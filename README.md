# audio2midi

A utility for converting an audio file into midi events.

## Training

1. Install the necessary pip dependencies:
```
jax
equinox
...
```

2. Install the Rust backed plugin:
  a. Install Rust (https://www.rust-lang.org/tools/install)
  b. `pip install maturin`
  c. Make sure you are in the desired Python venv
  d. Run `maturin develop -r` in the `rust-plugins` directory

3. Setup the dataset

4. Run `python train.py`

