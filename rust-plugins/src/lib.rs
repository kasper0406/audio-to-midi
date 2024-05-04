use std::path::Path;
use std::fmt;
use std::str::EncodeUtf16;

use numpy::{ PyArray2, PyArray3 };
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
use tokio::io::{BufReader, AsyncBufReadExt};
use once_cell::sync::Lazy;
use serde::Deserialize;
use uuid::Uuid;
use std::process::Stdio;
use futures::future::join_all;

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

fn key_to_event(key: u32) -> u32 {
    (key - 21) as u32
}

fn frame_position(time: f32, duration_per_frame: f32) -> u32 {
    (time / duration_per_frame).round() as u32
}

type MidiEvents = Vec<(u32, u32, u32, u32)>;
const VELOCITY_CATEGORIES: f32 = 10.0;

async fn get_events_from_file(path: &str, max_event_time: f32, duration_per_frame: f32) -> Result<MidiEvents, std::io::Error> {
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
                if record.time > max_event_time {
                    debug!["Skipping midi event because it happens after {} seconds", max_event_time];
                    continue
                }

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

    let mut events_with_padding: MidiEvents = vec![];
    events_with_padding.append(&mut events);
    Ok(events_with_padding)
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

async fn generate_raw_audio_using_ffmpeg(input_file: &str, left_output_file: &str, right_output_file: &str, sample_rate: u32, maybe_max_duration: Option<f32>) -> Result<(Vec<f32>, Vec<f32>), AudioLoadingError> {
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

    let left_samples = read_samples_from_file(left_output_file, sample_rate, maybe_max_duration);
    let right_samples = read_samples_from_file(right_output_file, sample_rate, maybe_max_duration);

    Ok((left_samples.await?, right_samples.await?))
}

async fn load_audio_sample(file_path: &str, sample_rate: u32, max_duration: Option<f32>) -> Result<(Vec<f32>, Vec<f32>), AudioLoadingError> {
    let uuid = Uuid::new_v4();
    let left_output_file = format!["/tmp/audio-to-midi-{}_left.raw", uuid];
    let right_output_file = format!["/tmp/audio-to-midi-{}_right.raw", uuid];
    
    let audio_sampels = generate_raw_audio_using_ffmpeg(file_path, &left_output_file, &right_output_file, sample_rate, max_duration).await;

    tokio::fs::remove_file(left_output_file.clone()).await
        .map_err(|err| AudioLoadingError::IoError(err, String::from(left_output_file)))?;
    tokio::fs::remove_file(right_output_file.clone()).await
        .map_err(|err| AudioLoadingError::IoError(err, String::from(right_output_file)))?;
    audio_sampels
}

#[pyfunction]
fn load_full_audio(py: Python, file: String, sample_rate: u32) -> PyResult<Py<PyArray2<f32>>> {
    let (left_samples, right_samples) = py.allow_threads(move || {
        TOKIO_RUNTIME.block_on(async {
            let future = TOKIO_RUNTIME.spawn(async move { load_audio_sample(&file, sample_rate, None).await });
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

fn convert_to_frame_events(events: &MidiEvents, frame_count: usize) -> Vec<Vec<f32>> {
    let num_event_types = 88;
    let base = 0.6;

    let mut frames = vec![vec![0.0; num_event_types]; frame_count];

    // Currently this is not a perfect representation, if a key is released and then attacked very quickly thereafter
    // but that will be an issue for another day...
    for (frame_start, key, frame_duration, velocity) in events {
        let velocity_boost = ((*velocity as f32) / VELOCITY_CATEGORIES) / 2.25;
        let start_strength = (base + velocity_boost).max(1.0);
        let decay_function = |t: f32| -> f32 {
            start_strength * (-0.05 * t).exp()
        };

        let frame_end = ((*frame_start + *frame_duration) as usize).min(frame_count);
        for frame in (*frame_start as usize)..frame_end {
            let t: f32 = frame as f32 - *frame_start as f32;
            frames[frame][*key as usize] = decay_function(t);
        }
    }

    frames
}

#[pyfunction]
fn load_events_and_audio(py: Python, dataset_dir: String, sample_names: &PyList, sample_rate: u32, max_duration: f32, duration_per_frame: f32) -> PyResult<(Py<PyList>, Py<PyList>, Py<PyList>)> {
    let sample_files = get_sample_files(py, dataset_dir, sample_names)?;

    let all_samples = py.allow_threads(move || {
        TOKIO_RUNTIME.block_on(async {
            let mut audio_futures = vec![];
            let mut event_futures = vec![];

            for sample_file in sample_files {
                let sampel_file_clone = sample_file.clone();

                let audio_future = TOKIO_RUNTIME.spawn(async move {
                    let audio_filename = resolve_audio_samples(&sampel_file_clone).await;
                    load_audio_sample(&audio_filename, sample_rate, Some(max_duration)).await
                });
                audio_futures.push(audio_future);

                let event_filename = format!["{}.csv", sample_file.clone()];
                let event_future = TOKIO_RUNTIME.spawn(async move { get_events_from_file(&event_filename, max_duration, duration_per_frame).await });
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

            let frame_count = (max_duration / duration_per_frame).round() as usize;
            let events_by_frame: Vec<Vec<Vec<f32>>> = events.iter()
                .map(|events| convert_to_frame_events(&events, frame_count))
                .collect();

            (audio_samples, events, events_by_frame)
        })
    });

    let mut audio_results = vec![];
    let mut event_results = vec![];
    let mut events_by_frame_results = vec![];
    let (all_audio_samples, all_events, events_by_frame) = all_samples;
    let iter = all_audio_samples.into_iter()
        .zip(all_events.into_iter())
        .zip(events_by_frame.into_iter())
        .map(|((a, b), c)| (a, b, c));
    for ((left_samples, right_samples), events, events_by_frame) in iter {
        let samples_vec = vec![left_samples, right_samples];
        let audio_array = PyArray2::from_vec2(py, &samples_vec)?.to_owned();
        audio_results.push(audio_array);

        let converted_events: Vec<Vec<u16>> = events.iter()
            .map(|(attack_time, key, duration, velocity)| vec![*attack_time as u16, *key as u16, *duration as u16, *velocity as u16])
            .collect();
        let event_array: Py<PyArray2<u16>> = PyArray2::from_vec2(py, &converted_events)?.to_owned();
        event_results.push(event_array);

        let converted_events_by_frame: Py<PyArray2<f32>> = PyArray2::from_vec2(py, &events_by_frame)?.to_owned();
        events_by_frame_results.push(converted_events_by_frame);
    }
    Ok((
        PyList::new(py, &audio_results).into(),
        PyList::new(py, &event_results).into(),
        PyList::new(py, &events_by_frame_results).into()
    ))
}

#[pymodule]
fn rust_plugins(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::init();

    m.add_function(wrap_pyfunction!(load_full_audio, m)?)?;
    m.add_function(wrap_pyfunction!(load_events_and_audio, m)?)?;
    Ok(())
}
