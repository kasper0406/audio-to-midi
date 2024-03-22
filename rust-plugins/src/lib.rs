use numpy::PyArray1;
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
    time: f64,
    duration: f64,
    key: u32,
    velocity: f64,
}

fn key_to_event(key: u32) -> u32 {
    2 + (key - 21)
}

async fn process_file(path: &str, max_event_time: f64) -> Result<String, std::io::Error> {
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
                if record.time < max_event_time {
                    events.push( (record.time, key_to_event(record.key) + 88, record.velocity) );
                }
                if record.time + record.duration < max_event_time + 5.0 * epsilon {
                    let release_time = max_event_time.min(record.time + record.duration - epsilon);
                    events.push( (release_time, key_to_event(record.key), 0.0) );
                }
            },
            Err(e) => eprintln!("Failed to deserialize record: {:?}", e),
        }
    }

    println!("Calculated events: {:?}", events);

    Ok(contents)
}

#[pyfunction]
fn events_from_samples(py: Python, dataset_dir: String, sample_names: &PyList, max_event_time: f64) -> PyResult<Py<PyArray1<f64>>> {
    TOKIO_RUNTIME.block_on(async {
        let mut join_set = JoinSet::new();

        for maybe_sample_name in sample_names {
            match maybe_sample_name.extract::<String>() {
                Ok(sample_name) => {
                    let sample_csv_file = format!("{}/{}.csv", dataset_dir, sample_name);
                    join_set.spawn(async move { process_file(&sample_csv_file, max_event_time).await });
                },
                Err(_) => {
                    return Err(PyTypeError::new_err("Sample names must be a list of strings!"));
                }
            }
        }

        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(Ok(content)) => {
                    println!("Read file content with length: {}", content.len());
                },
                Ok(Err(err)) => {
                    return Err(PyTypeError::new_err(format!("Failed to read sample: {:?}", err)))
                },
                Err(err) => {
                    return Err(PyTypeError::new_err(format!("Error during processing: {:?}", err)))
                }
            }
        }

        Ok(PyArray1::<f64>::zeros(py, 10, false).into())
    })
}

#[pymodule]
fn rust_plugins(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(events_from_samples, m)?)?;
    Ok(())
}
