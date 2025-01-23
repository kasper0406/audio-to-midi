use crate::common::MidiEvents;

use std::path::Path;
use std::fmt;
use std::str::EncodeUtf16;
use std::env;

use numpy::{ PyArray2, PyArray3, PyReadonlyArray2 };
use numpy::ToPyArray;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyList;
use pyo3::exceptions::PyTypeError;

use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::runtime::Runtime;
use tokio::process::Command;
use tokio::io::{BufReader, AsyncBufReadExt, AsyncWriteExt};
use once_cell::sync::Lazy;
use serde::Deserialize;
use uuid::Uuid;
use std::process::Stdio;
use futures::future::join_all;
use half::f16;
use num_traits::cast::AsPrimitive;
use futures::TryFutureExt;
use sha2::{Sha256, Digest};
use rand::prelude::*;
use rand_distr::{Distribution, Uniform, Normal, Beta};

use log::{debug, error, info, trace, warn};

static TOKIO_RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("Failed to create Tokio runtime")
});

#[derive(Debug, Deserialize)]
struct EventRecord {
    time: f32,
    duration: f32,
    key: u32,
    velocity: f32,
}

const NUM_EVENT_TYPES: usize = 90;

fn key_to_event(key: u32) -> u32 {
    (key - 21) as u32
}

fn frame_position(time: f32, duration_per_frame: f32) -> u32 {
    (time / duration_per_frame).round() as u32
}

const VELOCITY_CATEGORIES: f32 = 10.0;

async fn get_events_from_file(path: &str, duration_per_frame: f32) -> Result<MidiEvents, std::io::Error> {
    let mut file = File::open(path).await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .comment(Some(b'%'))
        .trim(csv::Trim::All)
        .from_reader(contents.as_bytes());
    reader.set_headers(csv::StringRecord::from(vec!["time", "duration", "key", "velocity"]));

    let mut events = vec![];
    for result in reader.deserialize::<EventRecord>().skip(1) {
        match result {
            Ok(record) => {
                let attack_time = frame_position(record.time, duration_per_frame);
                let key = key_to_event(record.key);
                let duration = frame_position(record.duration, duration_per_frame).max(1);
                let velocity = (record.velocity * (VELOCITY_CATEGORIES as f32)).round() as u32;

                events.push( (attack_time, key, duration, velocity) );
            },
            Err(e) => eprintln!("Failed to deserialize record: {:?}", e),
        }
    }
    events.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    Ok(events)
}

fn get_sample_files(py: Python, dataset_dir: String, sample_names: &PyList) -> Result<Vec<String>, PyErr> {
    let mut sample_files = vec![];
    for maybe_sample_name in sample_names {
        match maybe_sample_name.extract::<String>() {
            Ok(sample_name) => {
                let sample_csv_file = format!("{}/{}", dataset_dir, sample_name);
                sample_files.push(sample_csv_file);
            },
            Err(_) => {
                return Err(PyTypeError::new_err("Sample names must be a list of strings!"));
            }
        }
    }
    Ok(sample_files)
}

#[derive(Debug)]
enum AudioLoadingError {
    IoError(std::io::Error, String),
    FfmpegError(i32),
    ParseSampleRateError(),
}
impl std::error::Error for AudioLoadingError {}
impl fmt::Display for AudioLoadingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AudioLoadingError::IoError(err, file) => write!(f, "IO error: {}, file: {}", err, file),
            AudioLoadingError::FfmpegError(status) => write!(f, "ffmpeg terminated with status: {}", status),
            AudioLoadingError::ParseSampleRateError() => write!(f, "failed to parse ffprobe sample rate"),
        }
    }
}

fn is_aac_file(input_file: &str) -> bool {
    input_file.ends_with(".aac")
}

async fn get_aac_sample_rate(input_file: &str) -> Result<f64, AudioLoadingError> {
    let mut process = Command::new("ffprobe")
        .arg(input_file)
        .arg("-show_streams")
        .arg("-show_entries")
        .arg("stream=sample_rate")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg("-v")
        .arg("quiet")
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|err| AudioLoadingError::IoError(err, String::from(input_file)))?;

    let stdout = process.stdout.take().expect("child did not have a handle to stdout");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    reader.read_line(&mut line).await.map_err(|err| AudioLoadingError::IoError(err, String::from(input_file)))?;

    let sample_rate: u32 =  line.trim().parse().map_err(|_err| AudioLoadingError::ParseSampleRateError())?;

    process.wait().await.map_err(|err| AudioLoadingError::IoError(err, String::from(input_file)))?;
    debug!["Obtained sample rate {} for {}", sample_rate, input_file];
    Ok(sample_rate as f64)
}

async fn read_samples_from_file(raw_file: &str, sample_rate: u32, maybe_max_duration: Option<f32>) -> Result<Vec<f32>, AudioLoadingError> {
    let mut file = File::open(raw_file).await.map_err(|err| AudioLoadingError::IoError(err, String::from(raw_file)))?;
    let mut bytes = vec![];
    file.read_to_end(&mut bytes).await.map_err(|err| AudioLoadingError::IoError(err, String::from(raw_file)))?;

    let mut samples: Vec<_> = bytes.chunks_exact(4)
        .map(|chunk| {
            let buf: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(buf)
        })
        .collect();

    if let Some(max_duration) = maybe_max_duration {
        // Ensure the number of samples is exactly max_duration * sample_rate!
        let desired_sample_count = ((sample_rate as f32) * max_duration) as usize;
        samples.resize(desired_sample_count, 0.0);
    }

    Ok(samples)
}

async fn generate_raw_audio_using_ffmpeg(input_file: &str, left_output_file: &str, right_output_file: &str, sample_rate: u32, maybe_max_duration: Option<f32>) -> Result<(Vec<f16>, Vec<f16>), AudioLoadingError> {
    // TODO: Consider sending back all audio channels
    let mut command = Command::new("ffmpeg");
    if let Some(max_duration) = maybe_max_duration {
        command.arg("-t").arg(format!["{}", max_duration]);
    }

    let mut ffmpeg_audio_filter = String::from("[0:a]channelsplit=channel_layout=stereo[left][right]");
    if is_aac_file(input_file) {
        debug!["Processing aac file {}", input_file];
        // Unfortunately aac conversion using ffmpeg adds a delay to the beginning of the audio
        // I have found no easy way of removing this, so this is a hack to do it...
        command.arg("-c:a").arg("aac");

        let sample_rate = get_aac_sample_rate(input_file).await?;
        let delay = (2 * 1024) as f64 / sample_rate; // 2 AAC frames of 1024 samples over the sample rate
        debug!["Detecting AAC file, correcting with delay {}", delay];
        ffmpeg_audio_filter.push_str(&format!("; [left]atrim=start={}[left]; [right]atrim=start={}[right]", delay, delay));
        debug!["Filter after: {}", ffmpeg_audio_filter];
    }

    command.arg("-i")
        .arg(input_file)
        .arg("-filter_complex")
        .arg(ffmpeg_audio_filter)

         // Left channel mapping
        .arg("-map")
        .arg("[left]")
        .arg("-ar")
        .arg(format!["{}", sample_rate])
        .arg("-f")
        .arg("f32le")
        .arg(left_output_file)

        // Right channel mapping
        .arg("-map")
        .arg("[right]")
        .arg("-ar")
        .arg(format!["{}", sample_rate])
        .arg("-f")
        .arg("f32le")
        .arg(right_output_file)

        .stdout(Stdio::null())
        .stderr(Stdio::null());

    match command.status().await {
        Ok(status) => {
            if !status.success() {
                return Err(AudioLoadingError::FfmpegError(status.code().unwrap_or(-1)));
            }
        }
        Err(err) => return Err(AudioLoadingError::IoError(err, String::from(input_file))),
    }

    let left_samples_future = read_samples_from_file(left_output_file, sample_rate, maybe_max_duration);
    let right_samples_future = read_samples_from_file(right_output_file, sample_rate, maybe_max_duration);

    let left_samples = left_samples_future.await?;
    let right_samples = right_samples_future.await?;

    // Normalize the audio samples and convert to fp16 precision
    let max_value_left = left_samples.iter().map(|sample| sample.abs()).reduce(f32::max).unwrap();
    let max_value_right = right_samples.iter().map(|sample| sample.abs()).reduce(f32::max).unwrap();
    let mut total_max = max_value_left.max(max_value_right);
    if total_max <= 0.05 {
        // If the audio is very quite, there is a good chance it is silent everywhere, or it is simply noise
        // We do not want to normalize this due to division by 0 concerns
        let left_samples: Vec<f16> = left_samples.iter()
            .map(|sample| f16::from_f32(*sample))
            .collect();
        let right_samples: Vec<f16> = right_samples.iter()
            .map(|sample| f16::from_f32(*sample))
            .collect();

        Ok((left_samples, right_samples))
    } else {
        let total_elements = (left_samples.len() + right_samples.len()) as f64;
        let variance = left_samples.iter().zip(right_samples.iter())
            .fold(0f64, |acc, (left, right)| acc + (*left as f64).powi(2) + (*right as f64).powi(2)) / total_elements;
        let adjustment = (1.0 / variance).sqrt();

        let normalized_left: Vec<f16> = left_samples.iter()
            .map(|sample| f16::from_f64((*sample as f64) * adjustment))
            .collect();
        let normalized_right: Vec<f16> = right_samples.iter()
            .map(|sample| f16::from_f64((*sample as f64) * adjustment))
            .collect();

        Ok((normalized_left, normalized_right))
    }

}

async fn load_audio_sample_uncached(file_path: &str, sample_rate: u32) -> Result<(Vec<f16>, Vec<f16>), AudioLoadingError> {
    let uuid = Uuid::new_v4();
    let left_output_file = format!["/tmp/audio-to-midi-{}_left.raw", uuid];
    let right_output_file = format!["/tmp/audio-to-midi-{}_right.raw", uuid];
    
    let audio_sampels = generate_raw_audio_using_ffmpeg(file_path, &left_output_file, &right_output_file, sample_rate, None).await;

    tokio::fs::remove_file(left_output_file.clone()).await
        .map_err(|err| AudioLoadingError::IoError(err, String::from(left_output_file)))?;
    tokio::fs::remove_file(right_output_file.clone()).await
        .map_err(|err| AudioLoadingError::IoError(err, String::from(right_output_file)))?;
    audio_sampels
}

fn generate_cache_filename(path: &str, sample_rate: u32) -> String {
    // Hashing the complete path to ensure uniqueness
    let mut hasher = Sha256::new();
    hasher.update(path);
    let hash_result = hasher.finalize();
    let hash_str = format!("{:x}", hash_result); // Convert hash to hex string

    // Truncate hash to first 30 characters for brevity
    let short_hash = &hash_str[..30];

    // Combine truncated hash with filename
    format!("{}_{}", short_hash, sample_rate)
}

fn path_to_string_lossy<P: AsRef<Path>>(path: P) -> String {
    path.as_ref().to_string_lossy().into_owned()
}

async fn load_audio_sample(file_path: &str, sample_rate: u32, skip_cache: bool) -> Result<(Vec<f32>, Vec<f32>), AudioLoadingError> {
    match env::var("SAMPLE_CACHE_DIR") {
        Ok(cache_dir) => {
            let cache_filename = generate_cache_filename(file_path, sample_rate);
            let left_cache_file = Path::new(&cache_dir).join(&cache_filename[..4]).join(format!["{}_left.raw", cache_filename]);
            let right_cache_file = Path::new(&cache_dir).join(&cache_filename[..4]).join(format!["{}_right.raw", cache_filename]);

            tokio::fs::create_dir_all(left_cache_file.parent().unwrap()).await
                .map_err(|err| AudioLoadingError::IoError(err, path_to_string_lossy(&left_cache_file)))?;
            tokio::fs::create_dir_all(right_cache_file.parent().unwrap()).await
                .map_err(|err| AudioLoadingError::IoError(err, path_to_string_lossy(&right_cache_file)))?;

            if left_cache_file.exists() && right_cache_file.exists() && !skip_cache {
                debug!["Reading samples from cache {} {}", file_path, sample_rate];

                // Read the cached files
                let mut encoded_left = Vec::new();
                File::open(&left_cache_file).await
                    .map_err(|err| AudioLoadingError::IoError(err, path_to_string_lossy(&left_cache_file)))?
                    .read_to_end(&mut encoded_left).await
                    .map_err(|err| AudioLoadingError::IoError(err, path_to_string_lossy(&left_cache_file)))?;
                let maybe_left_samples: Result<Vec<f16>, _> = bincode::deserialize(&encoded_left[..]);

                let mut encoded_right = Vec::new();
                File::open(&right_cache_file).await
                    .map_err(|err| AudioLoadingError::IoError(err, path_to_string_lossy(&right_cache_file)))?
                    .read_to_end(&mut encoded_right).await
                    .map_err(|err| AudioLoadingError::IoError(err, path_to_string_lossy(&right_cache_file)))?;
                let maybe_right_samples: Result<Vec<f16>, _> = bincode::deserialize(&encoded_right[..]);

                if let Ok(left_samples) = maybe_left_samples {
                    if let Ok(right_samples) = maybe_right_samples {
                        return Ok((
                            left_samples.iter().map(|sample| sample.to_f32()).collect(),
                            right_samples.iter().map(|sample| sample.to_f32()).collect()
                        ))
                    }
                }

                tokio::fs::remove_file(left_cache_file.clone()).await
                    .map_err(|err| AudioLoadingError::IoError(err, path_to_string_lossy(&left_cache_file)))?;
                tokio::fs::remove_file(right_cache_file.clone()).await
                    .map_err(|err| AudioLoadingError::IoError(err, path_to_string_lossy(&left_cache_file)))?;
                return Box::pin(load_audio_sample(file_path, sample_rate, true)).await
            } else {
                let (left_samples, right_samples) = load_audio_sample_uncached(file_path, sample_rate).await?;

                // Update the cache
                File::create(&left_cache_file).await
                    .map_err(|err| AudioLoadingError::IoError(err, path_to_string_lossy(&left_cache_file)))?
                    .write_all(&bincode::serialize(&left_samples).unwrap()).await
                    .map_err(|err| AudioLoadingError::IoError(err, path_to_string_lossy(&left_cache_file)))?;
                File::create(&right_cache_file).await
                    .map_err(|err| AudioLoadingError::IoError(err, path_to_string_lossy(&right_cache_file)))?
                    .write_all(&bincode::serialize(&right_samples).unwrap()).await
                    .map_err(|err| AudioLoadingError::IoError(err, path_to_string_lossy(&left_cache_file)))?;

                Ok((
                    left_samples.iter().map(|sample| sample.to_f32()).collect(),
                    right_samples.iter().map(|sample| sample.to_f32()).collect()
                ))
            }
        },
        Err(e) => {
            let (left_samples, right_samples) = load_audio_sample_uncached(file_path, sample_rate).await?;
            Ok((
                left_samples.iter().map(|sample| sample.to_f32()).collect(),
                right_samples.iter().map(|sample| sample.to_f32()).collect()
            ))
        }
    }
}

#[pyfunction]
fn load_full_audio(py: Python, file: String, sample_rate: u32) -> PyResult<Py<PyArray2<f32>>> {
    let (left_samples, right_samples) = py.allow_threads(move || {
        TOKIO_RUNTIME.block_on(async {
            let future = TOKIO_RUNTIME.spawn(async move { load_audio_sample(&file, sample_rate, true).await });
            match future.await {
                Ok(Ok(samples)) => {
                    Ok(samples)
                },
                Ok(Err(err)) => {
                    Err(PyTypeError::new_err(format!("Failed to read sample: {:?}", err)))
                },
                Err(err) => {
                    Err(PyTypeError::new_err(format!("Error during processing: {:?}", err)))
                }
            }
        })
    }).unwrap();

    let samples_vec = vec![left_samples, right_samples];
    Ok(PyArray2::from_vec2(py, &samples_vec)?.to_owned())
}

async fn file_exists(file: &str) -> bool {
    match tokio::fs::metadata(file).await {
        Ok(metadata) => {
            true
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                false
            } else {
                error!["Failed to check file existence: {}", file];
                false
            }
        }
    }
}

async fn resolve_audio_samples(sample_file: &str) -> String {
    // This is kind of sucky, but ¯\_(ツ)_/¯
    if file_exists(&format!["{}.aac", sample_file]).await {
        return format!["{}.aac", sample_file]
    }
    if file_exists(&format!["{}.aif", sample_file]).await {
        return format!["{}.aif", sample_file]
    }
    panic!["Audio not found for sample: {}", sample_file];
}

fn convert_to_frame_events(events: &MidiEvents, model_output_size: i32, start_frame: i32, num_frames_with_backing_samples: i32) -> Vec<Vec<f32>> {
    let mut frames = vec![vec![0.0; NUM_EVENT_TYPES]; model_output_size as usize];

    for (attack_frame, key, frame_duration, velocity) in events {
        let decay_function = |t: f32| -> f32 {
            (-0.1 * t).exp().max(0.3) // Do not drop below 0.3 while the note is actually playing
        };

        let frame_start = (*attack_frame as i32) - start_frame;
        let frame_end = frame_start + *frame_duration as i32;

        // Ensure that the frame before the start is blank
        // To ensure we handle potentially fast re-activations
        if frame_start > 0 && frame_start < model_output_size {
            frames[(frame_start - 1) as usize][*key as usize] = 0.0;
        }

        for frame in frame_start.max(0)..frame_end.min(model_output_size).min(num_frames_with_backing_samples) {
            let t = frame as f32 - frame_start as f32;
            frames[frame as usize][*key as usize] = decay_function(t);
        }
    }

    frames
}

struct EventsAndAudio {
    samples: Vec<(Vec<f32>, Vec<f32>)>,
    events: Vec<Vec<Vec<f32>>>,
    sample_names: Vec<String>,
}

async fn load_events_and_audio_rust(sample_files: &[String], sample_rate: u32, model_duration: f32, num_model_outputs: i32, skip_cache: bool) -> EventsAndAudio {
    let duration_per_frame = model_duration as f64 / num_model_outputs as f64;

    let mut audio_futures = vec![];
    let mut event_futures = vec![];

    for sample_file in sample_files {
        let sampel_file_clone = sample_file.clone();

        let audio_future = TOKIO_RUNTIME.spawn(async move {
            let audio_filename = resolve_audio_samples(&sampel_file_clone).await;
            load_audio_sample(&audio_filename, sample_rate, skip_cache).await
        });
        audio_futures.push(audio_future);

        let event_filename = format!["{}.csv", sample_file.clone()];
        let event_future = TOKIO_RUNTIME.spawn(async move { get_events_from_file(&event_filename, duration_per_frame as f32).await });
        event_futures.push(event_future);
    }

    let audio_samples: Vec<(Vec<f32>, Vec<f32>)> = join_all(audio_futures).await
        .into_iter()
        .map(|result| result.unwrap())
        .map(|result| result.unwrap())
        .collect();
    let events: Vec<MidiEvents> = join_all(event_futures).await
        .into_iter()
        .map(|result| result.unwrap())
        .map(|result| result.unwrap())
        .collect();

    debug!("Event frame count {}, max_duration = {}, dpf = {}", num_model_outputs, model_duration, duration_per_frame);
    // TODO: Try to make this nicer...
    let (audio_samples, events_by_frame, sample_names) = events.iter()
            .zip(audio_samples.iter())
            .zip(sample_files.iter())
        .map(|((events, samples), sample_file_name)| {
            // events and samples may be too big for the model to handle. We will split them out
            let (left_samples, right_samples) = samples;
            let samples_per_model_call = (sample_rate as f32 * model_duration) as usize;
            let num_splits = ((left_samples.len() as f32) / samples_per_model_call as f32).ceil() as i32;

            let mut sample_splits = vec![];
            let mut event_splits = vec![];
            let mut sample_name_split = vec![];
            for split in 0..num_splits {
                let start_frame = split * num_model_outputs as i32;
                let start_sample = (split * samples_per_model_call as i32) as usize;
                let samples_to_copy = samples_per_model_call.min(left_samples.len() - start_sample);
                let num_frames_with_backing_samples = ((samples_to_copy as f32 / samples_per_model_call as f32) * (num_model_outputs as f32)).ceil() as i32;
                debug!["Copying {} samples starting at {} for split {}", samples_to_copy, start_sample, split];

                let frame_events = convert_to_frame_events(&events, num_model_outputs, start_frame, num_frames_with_backing_samples);
                let mut split_samples_left = vec![0.0; samples_per_model_call as usize];
                let mut split_samples_right = vec![0.0; samples_per_model_call as usize];

                for i in 0..samples_to_copy {
                    split_samples_left[i] = left_samples[start_sample + i];
                    split_samples_right[i] = right_samples[start_sample + i];
                }

                // Only include it if we have more than half of the samples so we do not train on too much silence
                if samples_to_copy > samples_per_model_call / 2 {
                    sample_splits.push((split_samples_left, split_samples_right));
                    event_splits.push(frame_events);
                    sample_name_split.push(format!["{}+{}", sample_file_name, split]);
                }
            }

            (sample_splits, event_splits, sample_name_split)
        })
        .fold((Vec::new(), Vec::new(), Vec::new()), |(mut acc_a, mut acc_b, mut acc_c), (a, b, c)| {
            acc_a.extend(a.into_iter().collect::<Vec<_>>());
            acc_b.extend(b.into_iter().collect::<Vec<_>>());
            acc_c.extend(c.into_iter().collect::<Vec<_>>());
            (acc_a, acc_b, acc_c)
        });

    EventsAndAudio {
        samples: audio_samples,
        events: events_by_frame,
        sample_names: sample_names,
    }
}

fn audio_and_samples_to_python(py: Python, events_and_audio: EventsAndAudio) -> PyResult<(Py<PyList>, Py<PyList>, Py<PyList>)> {
    let mut audio_results = vec![];
    let mut events_by_frame_results = vec![];
    let mut sample_name_results = vec![];
    let (all_audio_samples, events_by_frame, sample_names) = (events_and_audio.samples, events_and_audio.events, events_and_audio.sample_names);
    let iter = all_audio_samples.into_iter()
        .zip(events_by_frame.into_iter())
        .zip(sample_names.into_iter())
        .map(|((((a, b), c), d))| (a, b, c, d));
    for (left_samples, right_samples, events_by_frame, sample_name) in iter {
        let samples_vec = vec![left_samples, right_samples];
        let audio_array = PyArray2::from_vec2(py, &samples_vec)?.to_owned();
        audio_results.push(audio_array);

        let converted_events_by_frame: Py<PyArray2<f32>> = PyArray2::from_vec2(py, &events_by_frame)?.to_owned();
        events_by_frame_results.push(converted_events_by_frame);

        sample_name_results.push(sample_name);
    }
    Ok((
        PyList::new(py, &audio_results).into(),
        PyList::new(py, &events_by_frame_results).into(),
        PyList::new(py, &sample_name_results).into()
    ))
}

fn cut_mix_transformation(events_and_audio: &mut EventsAndAudio, cut_probability: f64) {
    let size = events_and_audio.samples.len();

    let sample_range = Uniform::from(0..size);
    let mut rng = rand::thread_rng();

    for _nr in 0..(cut_probability * (size as f64)) as usize {
        let a = sample_range.sample(&mut rng);
        let b = sample_range.sample(&mut rng);

        let min_cut = 0.01;
        let cut_start = Uniform::from(0.0..(1.0 - min_cut)).sample(&mut rng);
        let cut_length = Uniform::from(min_cut..(1.0 - cut_start)).sample(&mut rng);

        debug!["Applying CutMix from at (a, b) = ({}, {}) from {} -> {}", a, b, cut_start, cut_start + cut_length];

        let num_samples = events_and_audio.samples[a].0.len() as f64;
        let audio_cut_start = (cut_start * num_samples) as usize;
        let audio_cut_end = ((cut_start + cut_length) * num_samples) as usize;
    
        let (b_samples_left, b_samples_right) = {
            let (samples_left, samples_right) = &events_and_audio.samples[b];
            (
                samples_left[audio_cut_start..audio_cut_end].to_vec(),
                samples_right[audio_cut_start..audio_cut_end].to_vec(),
            )
        };

        let num_frames = events_and_audio.events[a].len() as f64;
        let event_cut_start = (cut_start * num_frames) as usize;
        let event_cut_end = ((cut_start + cut_length) * num_frames) as usize;

        let events_to_copy = {
            let copy_len = event_cut_end - event_cut_start;
            let num_events = events_and_audio.events[a][0].len();
            let mut events_to_copy = vec![vec![0.0; num_events]; copy_len];

            for i in event_cut_start..event_cut_end {
                for event_idx in 0..events_and_audio.events[a][i].len() {
                    events_to_copy[i - event_cut_start][event_idx] = events_and_audio.events[b][i][event_idx];
                }
            }
            events_to_copy
        };

        let (ref mut left_channel_a, ref mut right_channel_a) = &mut events_and_audio.samples[a];
        for i in audio_cut_start..audio_cut_end {
            left_channel_a[i] = b_samples_left[i - audio_cut_start];
            right_channel_a[i] = b_samples_right[i - audio_cut_start];
        }

        for i in event_cut_start..event_cut_end {
            for event_idx in 0..events_and_audio.events[a][i].len() {
                events_and_audio.events[a][i][event_idx] = events_to_copy[i - event_cut_start][event_idx];
            }
        }
    }
}

fn mixup_transformation(events_and_audio: &mut EventsAndAudio, mixup_probability: f64) {
    let size = events_and_audio.samples.len();

    let sample_range = Uniform::from(0..size);
    let mut rng = rand::thread_rng();

    for _nr in 0..(mixup_probability * (size as f64)) as usize {
        let a = sample_range.sample(&mut rng);
        let b = sample_range.sample(&mut rng);

        let lambda = Beta::new(2.0, 2.0).unwrap().sample(&mut rng);

        debug!["Applying MixUp at (a, b) = ({}, {}) with lambda = {}", a, b, lambda];

        let (new_samples_left, new_samples_right) = {
            let (a_samples_left, a_samples_right) = &events_and_audio.samples[a];
            let (b_samples_left, b_samples_right) = &events_and_audio.samples[b];

            let mut new_samples_left = vec![0.0; a_samples_left.len()];
            let mut new_samples_right = vec![0.0; a_samples_right.len()];

            for i in 0..a_samples_left.len() {
                new_samples_left[i] = lambda * a_samples_left[i] + (1.0 - lambda) * b_samples_left[i];
                new_samples_right[i] = lambda * a_samples_right[i] + (1.0 - lambda) * b_samples_right[i];
            }

            (new_samples_left, new_samples_right)
        };

        let num_frames = events_and_audio.events[a].len();
        let events_to_copy = {
            let num_events = events_and_audio.events[a][0].len();
            let mut events_to_copy = vec![vec![0.0; num_events]; num_frames];

            for i in 0..num_frames {
                for event_idx in 0..events_and_audio.events[a][i].len() {
                    let event_val_a = events_and_audio.events[a][i][event_idx];
                    let event_val_b = events_and_audio.events[b][i][event_idx];
                    events_to_copy[i][event_idx] = event_val_a.max(event_val_b);
                }
            }
            events_to_copy
        };

        let (ref mut left_channel_a, ref mut right_channel_a) = &mut events_and_audio.samples[a];
        for i in 0..left_channel_a.len() {
            left_channel_a[i] = new_samples_left[i];
            right_channel_a[i] = new_samples_right[i];
        }

        for i in 0..num_frames {
            for event_idx in 0..events_and_audio.events[a][i].len() {
                events_and_audio.events[a][i][event_idx] = events_to_copy[i][event_idx];
            }
        }
    }
}

fn rotate_transformation(events_and_audio: &mut EventsAndAudio, rotate_probability: f64) {
    let size = events_and_audio.samples.len();

    let sample_range = Uniform::from(0..size);
    let mut rng = rand::thread_rng();

    for _nr in 0..(rotate_probability * (size as f64)) as usize {
        let idx = sample_range.sample(&mut rng);
        let roll_distance = Uniform::from(0.0..1.0).sample(&mut rng);
        debug!["Rotating idx {} by {}", idx, roll_distance];
        
        let num_samples = events_and_audio.samples[idx].0.len() as f64;
        let audio_rotations = (roll_distance * num_samples) as usize;
        let num_frames = events_and_audio.events[idx].len() as f64;
        let event_rotations = (roll_distance * num_frames) as usize;
        
        let (ref mut left_channel, ref mut right_channel) = &mut events_and_audio.samples[idx];
        left_channel.rotate_right(audio_rotations);
        right_channel.rotate_right(audio_rotations);

        events_and_audio.events[idx].rotate_right(event_rotations);
    }
}

fn random_erasing_transformation(events_and_audio: &mut EventsAndAudio, erase_probability: f64) {
    let size = events_and_audio.samples.len();

    let sample_range = Uniform::from(0..size);
    let mut rng = rand::thread_rng();

    let min_erase: f64 = 0.01;
    let max_erase: f64 = 0.10;

    for _nr in 0..(erase_probability * (size as f64)) as usize {
        let idx = sample_range.sample(&mut rng);
        let erase_start = Uniform::from(0.0..(1.0 - min_erase)).sample(&mut rng);
        let erase_length = Uniform::from(min_erase..max_erase.min(1.0 - erase_start)).sample(&mut rng);
        debug!["Erasing idx {} from {} -> {}", idx, erase_start, erase_start + erase_length];
        
        let num_samples = events_and_audio.samples[idx].0.len() as f64;
        let audio_erase_start = (erase_start * num_samples) as usize;
        let audio_erase_end = ((erase_start + erase_length) * num_samples) as usize;
        
        let (ref mut left_channel, ref mut right_channel) = &mut events_and_audio.samples[idx];
        for i in audio_erase_start..audio_erase_end {
            left_channel[i] = 0.0;
            right_channel[i] = 0.0;
        }
    }
}

fn gain_transformation(events_and_audio: &mut EventsAndAudio, gain_probability: f64) {
    let size = events_and_audio.samples.len();

    let sample_range = Uniform::from(0..size);
    let mut rng = rand::thread_rng();

    for _nr in 0..(gain_probability * (size as f64)) as usize {
        let idx = sample_range.sample(&mut rng);
        let gain = Normal::new(1.0f32, 0.25f32).unwrap().sample(&mut rng).min(1.5f32).max(0.5f32);
        debug!["Gain adjustment for {}: {}", idx, gain];
        
        let num_samples = events_and_audio.samples[idx].0.len() as f64;
        
        let (ref mut left_channel, ref mut right_channel) = &mut events_and_audio.samples[idx];
        for i in 0..left_channel.len() {
            left_channel[i] = left_channel[i] * gain;
            right_channel[i] = right_channel[i] * gain;
        }
    }
}

fn noise_transformation(events_and_audio: &mut EventsAndAudio, noise_probability: f64) {
    let size = events_and_audio.samples.len();

    let sample_range = Uniform::from(0..size);
    let mut rng = rand::thread_rng();

    for _nr in 0..(noise_probability * (size as f64)) as usize {
        let idx = sample_range.sample(&mut rng);
        let sigma = Uniform::new(0.0, 0.25).sample(&mut rng);
        let noise_distr = Normal::new(0.0f32, sigma).unwrap();
        debug!["Adding noise for {} with sigma = {}", idx, sigma];
        
        let num_samples = events_and_audio.samples[idx].0.len() as f64;
        
        let (ref mut left_channel, ref mut right_channel) = &mut events_and_audio.samples[idx];
        for i in 0..left_channel.len() {
            left_channel[i] = left_channel[i] + noise_distr.sample(&mut rng);
            right_channel[i] = right_channel[i] + noise_distr.sample(&mut rng);
        }
    }
}

fn label_smoothing_transformation(events_and_audio: &mut EventsAndAudio, alpha: f32) {
    let size = events_and_audio.samples.len();
    let sample_range = Uniform::from(0..size);

    for idx in 0..size {
        let num_frames = events_and_audio.events[idx].len();
        for i in 0..num_frames {
            for event_idx in 0..events_and_audio.events[idx][i].len() {
                let mut current_value = events_and_audio.events[idx][i][event_idx];
                let mut updated_value = current_value.min(1.0 - alpha).max(alpha);
                events_and_audio.events[idx][i][event_idx] = updated_value;
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
#[pyclass]
struct DatasetTransfromSettings {
    #[pyo3(get, set)]
    cut_probability: f64,

    #[pyo3(get, set)]
    rotate_probability: f64,

    #[pyo3(get, set)]
    random_erasing_probability: f64,

    #[pyo3(get, set)]
    mixup_probability: f64,

    #[pyo3(get, set)]
    gain_probability: f64,

    #[pyo3(get, set)]
    noise_probability: f64,

    #[pyo3(get, set)]
    label_smoothing_alpha: f32,
}

#[pymethods]
impl DatasetTransfromSettings {
    #[new]
    fn new(
        cut_probability: f64,
        rotate_probability: f64,
        random_erasing_probability: f64,
        mixup_probability: f64,
        gain_probability: f64,
        noise_probability: f64,
        label_smoothing_alpha: f32,
    ) -> Self {
        DatasetTransfromSettings {
            cut_probability,
            rotate_probability,
            random_erasing_probability,
            mixup_probability,
            gain_probability,
            noise_probability,
            label_smoothing_alpha,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetTransfromSettings(cut_probability={}, rotate_probability={}, random_erasing_probability={}, mixup_probability={}, gain_probability={}, noise_probability={}, label_smoothing_alpha={})",
            self.cut_probability,
            self.rotate_probability,
            self.random_erasing_probability,
            self.mixup_probability,
            self.gain_probability,
            self.noise_probability,
            self.label_smoothing_alpha
        )
    }
}

fn transform_for_training(mut events_and_audio: &mut EventsAndAudio, settings: &DatasetTransfromSettings) {
    cut_mix_transformation(&mut events_and_audio, settings.cut_probability);
    rotate_transformation(&mut events_and_audio, settings.rotate_probability);
    random_erasing_transformation(&mut events_and_audio, settings.random_erasing_probability);
    mixup_transformation(&mut events_and_audio, settings.mixup_probability);
    gain_transformation(&mut events_and_audio, settings.gain_probability);
    noise_transformation(&mut events_and_audio, settings.noise_probability);
    label_smoothing_transformation(&mut events_and_audio, settings.label_smoothing_alpha);
}

#[pyfunction]
fn load_events_and_audio_with_transformations(py: Python, dataset_dir: String, sample_names: &PyList, sample_rate: u32, model_duration: f32, num_model_outputs: i32, settings: DatasetTransfromSettings, skip_cache: bool) -> PyResult<(Py<PyList>, Py<PyList>, Py<PyList>)> {
    let sample_files = get_sample_files(py, dataset_dir, sample_names)?;

    let events_and_audio = py.allow_threads(move || {
        TOKIO_RUNTIME.block_on(async {
            let mut events_and_audio = load_events_and_audio_rust(&sample_files, sample_rate, model_duration, num_model_outputs, skip_cache).await;
            transform_for_training(&mut events_and_audio, &settings);
            return events_and_audio
        })
    });

    audio_and_samples_to_python(py, events_and_audio)
}

#[pyfunction]
fn load_events_and_audio(py: Python, dataset_dir: String, sample_names: &PyList, sample_rate: u32, model_duration: f32, num_model_outputs: i32, skip_cache: bool) -> PyResult<(Py<PyList>, Py<PyList>, Py<PyList>)> {
    let sample_files = get_sample_files(py, dataset_dir, sample_names)?;

    let events_and_audio = py.allow_threads(move || {
        TOKIO_RUNTIME.block_on(async {
            load_events_and_audio_rust(&sample_files, sample_rate, model_duration, num_model_outputs, skip_cache).await
        })
    });

    audio_and_samples_to_python(py, events_and_audio)
}

#[pyfunction]
fn stitch_probs(py: Python, py_probs: Py<PyArray3<f32>>, overlap: f64, duration_per_frame: f64) -> PyResult<Py<PyArray2<f32>>> {
    let array = py_probs.as_ref(py).readonly();
    let probs = array.as_array();

    let stitched_probs = crate::common::stitch_probs(&probs, overlap, duration_per_frame);
    Ok(stitched_probs.into_pyarray_bound(py).into())
}

#[pyfunction]
fn extract_events(py: Python, py_probs: Py<PyArray2<f32>>) -> PyResult<Py<PyList>> {
    let array = py_probs.as_ref(py).readonly();
    let probs = array.as_array();
    
    let events = crate::common::extract_events(&probs);
    Ok(PyList::new(py, &events).into())
}

#[pyfunction]
fn to_frame_events(py: Python, py_events: Py<PyList>, frame_count: usize) -> PyResult<Py<PyList>> {
    let all_events: Vec<_> = py_events.as_ref(py).iter()
        .map(|element| {
            let mut events = vec![];
            for event in element.iter().unwrap() {
                let event = event.unwrap();
                if let Ok((attack_time, key, duration, velocity)) = event.extract() {
                    events.push((attack_time, key, duration, velocity))
                } else {
                    eprintln!("Unknown event: {:?}", event);
                }
            }
            events
        })
        .collect();

    let converted: Vec<_> = all_events.iter()
        .map(|events| {
            let rust_converted = convert_to_frame_events(&events, frame_count as i32, 0, frame_count as i32);
            PyArray2::from_vec2(py, &rust_converted).unwrap().to_owned()
        })
        .collect();

    Ok(PyList::new(py, &converted).into())
}

#[pymodule]
fn modelutil(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::init();

    m.add_class::<DatasetTransfromSettings>()?;

    m.add_function(wrap_pyfunction!(load_full_audio, m)?)?;
    m.add_function(wrap_pyfunction!(load_events_and_audio, m)?)?;
    m.add_function(wrap_pyfunction!(load_events_and_audio_with_transformations, m)?)?;
    m.add_function(wrap_pyfunction!(stitch_probs, m)?)?;
    m.add_function(wrap_pyfunction!(extract_events, m)?)?;
    m.add_function(wrap_pyfunction!(to_frame_events, m)?)?;
    Ok(())
}
