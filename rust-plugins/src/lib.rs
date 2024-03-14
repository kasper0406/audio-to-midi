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

async fn process_file(path: &str, max_event_time: f32) -> Result<MidiEvents, std::io::Error> {
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
                    events.push( (record.time, key_to_event(record.key) + 88, record.velocity) );
                    max_velocity = max_velocity.max(record.velocity)
                }
                if record.time + record.duration < max_event_time + 5.0 * epsilon {
                    let release_time = max_event_time.min(record.time + record.duration - epsilon);
                    events.push( (release_time, key_to_event(record.key), 0.0) );
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

#[pyfunction]
fn events_from_samples(py: Python, dataset_dir: String, sample_names: &PyList, max_event_time: f32) -> PyResult<Py<PyList>> {
    let mut sample_files = vec![];
    for maybe_sample_name in sample_names {
        match maybe_sample_name.extract::<String>() {
            Ok(sample_name) => {
                let sample_csv_file = format!("{}/{}.csv", dataset_dir, sample_name);
                sample_files.push(sample_csv_file);
            },
            Err(_) => {
                return Err(PyTypeError::new_err("Sample names must be a list of strings!"));
            }
        }
    }

    let all_events = py.allow_threads(move || {
        TOKIO_RUNTIME.block_on(async {
            let mut futures = vec![];

            for sample_file in sample_files {
                let future = TOKIO_RUNTIME.spawn(async move { process_file(&sample_file, max_event_time).await });
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

#[pymodule]
fn rust_plugins(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(events_from_samples, m)?)?;
    Ok(())
}
