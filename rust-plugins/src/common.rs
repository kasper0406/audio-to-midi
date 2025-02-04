use ndarray::ArrayView2;
use ndarray::ArrayView3;
use ndarray::Array;
use ndarray::Array2;
use num_traits::Zero;

use num_traits::cast::AsPrimitive;

use log::debug;

pub type MidiEvents = Vec<(u32, u32, u32, u32)>;

pub fn stitch_probs<'a, T>(all_probs: &ArrayView3<'a, T>, overlap: f64, duration_per_frame: f64) -> Array2<f32>
where
    T: Zero + Clone + Copy + std::ops::Div<Output = T> + AsPrimitive<f32>
{
    let [num_windows, frames_per_window, event_categories] = *all_probs.shape() else { todo!() };
    let (num_windows, frames_per_window, event_categories) = (num_windows as i64, frames_per_window as i64, event_categories as i64);

    let overlapping_frames = overlap as f64 / duration_per_frame as f64;
    debug!("Overlapping frames: {}, overlap: {}, dpf: {}", overlapping_frames, overlap, duration_per_frame);
    let output_frames = (num_windows * frames_per_window - (overlapping_frames as i64) * (num_windows - 1)) as usize;
    let mut stitched: Array2<f32> = Array::zeros((output_frames, event_categories as usize));
    let mut output_frame_base = 0.0;
    for window in 0..num_windows {
        for frame in 0..frames_per_window {
            for event in 0..event_categories {
                let stitched_idx = (((output_frame_base as i64) + frame) as usize, event as usize);
                let probs_idx = (window as usize, frame as usize, event as usize);

                // If we are in the overlapping frames area, we need to blend the results
                if window > 0 && frame <= overlapping_frames.ceil() as i64 {
                    // We do a linear blend from 0 to 1 in the region of overlapping frames
                    let blend = (frame as f64) / (overlapping_frames as f64); // Blend will be in [0; 1]
                    stitched[stitched_idx] = ((1.0 - blend) * (stitched[stitched_idx] as f64) + blend * (all_probs[probs_idx].as_() as f64)) as f32;
                } else {
                    stitched[stitched_idx] = all_probs[probs_idx].as_();
                }
            }
        }
        output_frame_base += (frames_per_window as f64) - overlapping_frames;
    }

    stitched
}

pub fn extract_events<T>(probs: &ArrayView2<T>) -> MidiEvents
where
    T: AsPrimitive<f32>,
{
    let reactivation_threshold = 0.2 as f32;
    let activation_threshold = 0.4 as f32;
    let deactivation_threshold = 0.05 as f32;

    let mut events: MidiEvents = vec![];
    let [num_frames, num_notes] = *probs.shape() else { todo!("Unsupported probs format") };

    let duration = |end_frame: usize, start_frame: usize| -> u32 {
        ((end_frame as i32) - (start_frame as i32)).max(1) as u32
    };

    let velocity = |_activation_prob: f32| -> u32 {
        // TODO(knielsen): Implement this
        7
    };

    let mut currently_playing: Vec<Option<(usize, f32)>> = vec![None; num_notes];
    for frame in 0..num_frames {
        for key in 0..num_notes {
            let get_activation_prob = || -> f32 {
                let mut activation_prob = probs[(frame, key)].as_();
                let lookahead = 5;
                for i in (frame + 1)..num_frames {
                    if probs[(i, key)].as_() > activation_prob {
                        activation_prob = probs[(i, key)].as_();
                    } else {
                        if i - frame > lookahead {
                            break;
                        }
                    }
                }
                activation_prob
            };

            if let Some((started_at, activation_prob)) = currently_playing[key] {
                if probs[(frame, key)].as_() < deactivation_threshold {
                    // Handle case where a currently playing note stopped playing
                    events.push((started_at as u32, key as u32, duration(frame, started_at), velocity(activation_prob)));
                    currently_playing[key] = None;
                } else {
                    // Handle the case where there may be a re-activation
                    let time_since_activation = frame as f32 - started_at as f32;

                    // The way note re-activation has been implemented during training is quite implicit
                    // We expect the probability of a note to increase (instead of decrease) when a note is re-attacked
                    // We try to figure this out by computing the average probability of the past frames and the next frames
                    let mut should_reactivate = false;
                    if time_since_activation > 5.0 {
                        let samples = 5;
                        let mut prev_average = 0.0;
                        for i in (frame - samples)..frame {
                            prev_average += probs[(i, key)].as_();
                        }
                        prev_average /= samples as f32;

                        let mut next_average: f32 = 0.0;
                        for i in frame..(frame + samples).min(num_frames) {
                            next_average += probs[(i, key)].as_();
                        }
                        next_average /= samples as f32;

                        should_reactivate = next_average - prev_average > 0.05;
                    }

                    if frame < num_frames - 1 && probs[(frame, key)].as_() < probs[(frame + 1, key)].as_() {
                        // We handle the re-activation in the next frame where the probability is larger
                        continue;
                    }

                    if probs[(frame, key)].as_() > reactivation_threshold && should_reactivate {
                        events.push((started_at as u32, key as u32, duration(frame - 1, started_at), velocity(activation_prob))); // Close the old event
                        currently_playing[key] = Some((frame, get_activation_prob()));
                    }
                }
            } else {
                // The model output quite a bit of noise for bass notes. Only emit them if we are very confident
                let mut dimish_bass_notes = 0.0;
                if key < 30 {
                    dimish_bass_notes += ((30.0 - key as f32) / 30.0) * 0.3;
                }

                if probs[(frame, key)].as_() > activation_threshold + dimish_bass_notes {
                    currently_playing[key] = Some((frame, get_activation_prob()));
                }
            }
        }
    }

    // There may be currently playing events we need to meit
    for key in 0..num_notes {
        if let Some((started_at, activation_prob)) = currently_playing[key] {
            events.push((started_at as u32, key as u32, duration(num_frames, started_at), velocity(activation_prob)));
            currently_playing[key] = None;
        }
    }

    events.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    events
}
