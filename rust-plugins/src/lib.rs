use std::path::Path;
use std::fmt;

use numpy::PyArray1;
use numpy::PyArray2;
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
use once_cell::sync::Lazy;
use tokio::task::JoinSet;
use serde::Deserialize;
use uuid::Uuid;
use std::process::Stdio;

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
    FfmpegError(i32),
}
impl std::error::Error for AudioLoadingError {}
impl fmt::Display for AudioLoadingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AudioLoadingError::IoError(err) => write!(f, "IO error: {}", err),
            AudioLoadingError::FfmpegError(status) => write!(f, "ffmpeg terminated with status: {}", status),
        }
    }
}

async fn generate_raw_audio_using_ffmpeg(input_file: &str, output_file: &str, max_duration: f32) -> Result<Vec<f32>, AudioLoadingError> {
    let sample_rate = 16_000; // Hz

    let status = Command::new("ffmpeg")
        .arg("-i")
        .arg(input_file)
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg(format!["{}", sample_rate])
        .arg("-f")
        .arg("f32le")
        .arg(output_file)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
    match status {
        Ok(status) => {
            if !status.success() {
                return Err(AudioLoadingError::FfmpegError(status.code().unwrap_or(-1)));
            }
        }
        Err(err) => return Err(AudioLoadingError::IoError(err)),
    }

    let mut file = File::open(output_file).await.map_err(AudioLoadingError::IoError)?;
    let mut bytes = vec![];
    file.read_to_end(&mut bytes).await.map_err(AudioLoadingError::IoError)?;

    let mut samples: Vec<_> = bytes.chunks_exact(4)
        .map(|chunk| {
            let buf: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(buf)
        })
        .collect();

    // Ensure the number of samples is exactly max_duration * sample_rate!
    let desired_sample_count = ((sample_rate as f32) * max_duration) as usize;
    samples.resize(desired_sample_count, 0.0);

    Ok(samples)
}

async fn load_audio_sample(file_path: &str, max_duration: f32) -> Result<Vec<f32>, AudioLoadingError> {
    let path = Path::new(file_path);
    
    let uuid = Uuid::new_v4();
    let output_file = format!["/tmp/audio-to-midi-{}.raw", uuid];
    
    let audio_sampels = generate_raw_audio_using_ffmpeg(file_path, &output_file, max_duration).await;

    tokio::fs::remove_file(output_file).await.map_err(AudioLoadingError::IoError)?;
    audio_sampels
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
        numpy_results.push(samples.into_pyarray(py).to_owned());
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
