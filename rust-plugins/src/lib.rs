use std::path::Path;
use std::fmt;

use numpy::PyArray1;
use numpy::PyArray2;
use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyList;
use pyo3::exceptions::PyTypeError;

use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::runtime::Runtime;
use once_cell::sync::Lazy;
use tokio::task::JoinSet;
use serde::Deserialize;

use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

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
    2 + (key - 21) as u32
}

type MidiEvents = Vec<(f32, u32, f32)>;

async fn get_events_from_file(path: &str, max_event_time: f32) -> Result<MidiEvents, std::io::Error> {
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
    let mut max_velocity: f32 = 0.0;
    for result in reader.deserialize::<EventRecord>().skip(1) {
        match result {
            Ok(record) => {
                if record.time < max_event_time {
                    events.push( (record.time, key_to_event(record.key), record.velocity) );
                    max_velocity = max_velocity.max(record.velocity)
                }
                if record.time + record.duration < max_event_time {
                    events.push( (record.time + record.duration, key_to_event(record.key), 0.0) );
                }
            },
            Err(e) => eprintln!("Failed to deserialize record: {:?}", e),
        }
    }
    events.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));

    // Normalize the velocity. TODO: Refactor this
    if max_velocity != 0.0 {
        for i in 0..events.len() {
            let (time, key, velocity) = events[i];
            events[i] = (time, key, velocity / max_velocity);
        }
    }

    let mut events_with_padding: MidiEvents = vec![];
    events_with_padding.push((0.0, 1, 0.0));
    events_with_padding.append(&mut events);
    events_with_padding.push((0.0, 0, 0.0));
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

#[pyfunction]
fn events_from_samples(py: Python, dataset_dir: String, sample_names: &PyList, max_event_time: f32) -> PyResult<Py<PyList>> {
    let sample_files = get_sample_files(py, dataset_dir, sample_names)?;

    let all_events = py.allow_threads(move || {
        TOKIO_RUNTIME.block_on(async {
            let mut futures = vec![];

            for sample_file in sample_files {
                let future = TOKIO_RUNTIME.spawn(async move { get_events_from_file(&format!["{}.csv", sample_file], max_event_time).await });
                futures.push(future);
            }

            let mut all_events = vec![];
            for future in futures {
                match future.await {
                    Ok(Ok(events)) => {
                        all_events.push(events);
                    },
                    Ok(Err(err)) => {
                        return Err(PyTypeError::new_err(format!("Failed to read sample: {:?}", err)))
                    },
                    Err(err) => {
                        return Err(PyTypeError::new_err(format!("Error during processing: {:?}", err)))
                    }
                }
            }
            Ok(all_events)
        })
    }).unwrap();

    let mut numpy_results = vec![];
    for events in all_events {
        let mut np_array = PyArray2::<f32>::zeros(py, [events.len(), 3], false);
        {
            let mut array_read_write = np_array.readwrite();
            for (row, (time, key, velocity)) in events.into_iter().enumerate() {
                *array_read_write.get_mut([row, 0]).unwrap() = time as f32;
                *array_read_write.get_mut([row, 1]).unwrap() = key as f32;
                *array_read_write.get_mut([row, 2]).unwrap() = velocity as f32;
            }
        }
        numpy_results.push(np_array);
    }
    Ok(PyList::new(py, &numpy_results).into())
}

#[derive(Debug)]
enum AudioLoadingError {
    IoError(std::io::Error),
    AudioLoadingError(symphonia::core::errors::Error),
    CodecNotFound(),
}
impl std::error::Error for AudioLoadingError {}
impl fmt::Display for AudioLoadingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AudioLoadingError::IoError(err) => write!(f, "IO error: {}", err),
            AudioLoadingError::AudioLoadingError(err) => write!(f, "Audio processing error: {}", err),
            AudioLoadingError::CodecNotFound() => write!(f, "Did not find a suitable audio codec"),
        }
    }
}

async fn load_audio_sample(file_path: &str, max_duration: f32) -> Result<Vec<f32>, AudioLoadingError> {
    let path = Path::new(file_path);
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    // Open the sample file and read the sample
    // let mut file = File::open(path).await.map_err(AudioLoadingError::IoError)?;
    // let mut buffer = Vec::new();
    // file.read_to_end(&mut buffer).await.map_err(AudioLoadingError::IoError)?;

    // // Decode the audio and get the samples
    // use std::io::Cursor;
    // let mss = MediaSourceStream::new(Box::new(Cursor::new(buffer)), Default::default());

    let file = Box::new(std::fs::File::open(Path::new(&file_path)).unwrap());
    let mss = MediaSourceStream::new(file, Default::default());

    let mut hint = Hint::new();
    hint.with_extension(extension);

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    debug!("Using extension hint {}", extension);

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .map_err(AudioLoadingError::AudioLoadingError)?;
    
    debug!("bar!");

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(AudioLoadingError::CodecNotFound)?;
    let track_id = track.id;

    debug!("Found a track!");

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .map_err(AudioLoadingError::AudioLoadingError)?;

    debug!("Found a decoder!");

    loop {
        // Get the next packet from the media format.
        let packet = format.next_packet().map_err(AudioLoadingError::AudioLoadingError)?;
        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples.
        match decoder.decode(&packet) {
            Ok(decoded) => {
                info!("Decoded {} bytes", decoded.capacity());
            }
            Err(Error::IoError(err)) => {
                // The packet failed to decode due to an IO error, skip the packet.
                warn!("Skipping package: {}", err);
                continue;
            }
            Err(Error::DecodeError(err)) => {
                // The packet failed to decode due to invalid data, skip the packet.
                warn!("Skipping package: {}", err);
                continue;
            }
            Err(err) => {
                return Err(AudioLoadingError::AudioLoadingError(err));
            }
        }
    }

    Ok(vec![])
}

#[pyfunction]
fn load_audio_samples(py: Python, dataset_dir: String, sample_names: &PyList, max_duration: f32) -> PyResult<Py<PyList>> {
    let sample_files = get_sample_files(py, dataset_dir, sample_names)?;

    let all_samples = py.allow_threads(move || {
        TOKIO_RUNTIME.block_on(async {
            let mut futures = vec![];

            for sample_file in sample_files {
                let filename = format!["{}.aac", sample_file];
                let future = TOKIO_RUNTIME.spawn(async move { load_audio_sample(&filename, max_duration).await });
                futures.push(future);
            }

            let mut all_samples = vec![];
            for future in futures {
                match future.await {
                    Ok(Ok(samples)) => {
                        all_samples.push(samples);
                    },
                    Ok(Err(err)) => {
                        return Err(PyTypeError::new_err(format!("Failed to read sample: {:?}", err)))
                    },
                    Err(err) => {
                        return Err(PyTypeError::new_err(format!("Error during processing: {:?}", err)))
                    }
                }
            }
            Ok(all_samples)
        })
    }).unwrap();

    let mut numpy_results = vec![];
    for samples in all_samples {
        let mut np_array = PyArray1::<f32>::zeros(py, samples.len(), false);
        {
            let mut array_read_write = np_array.readwrite();
            for i in 0..samples.len() {
                *array_read_write.get_mut(i).unwrap() = samples[i];
            }
        }
        numpy_results.push(np_array);
    }
    Ok(PyList::new(py, &numpy_results).into())
}

#[pymodule]
fn rust_plugins(py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::init();

    m.add_function(wrap_pyfunction!(events_from_samples, m)?)?;
    m.add_function(wrap_pyfunction!(load_audio_samples, m)?)?;
    Ok(())
}
