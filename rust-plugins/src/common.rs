use ndarray::ArrayView2;

use num_traits::cast::AsPrimitive;

pub type MidiEvents = Vec<(u32, u32, u32, u32)>;

pub fn extract_events<T>(probs: &ArrayView2<T>) -> MidiEvents
where
    T: AsPrimitive<f32>,
{
    let activation_threshold = 0.5 as f32;
    let deactivation_threshold = 0.1 as f32;

    let mut events: MidiEvents = vec![];
    let [num_frames, num_notes] = *probs.shape() else { todo!("Unsupported probs format") };

    let duration = |end_frame: usize, start_frame: usize| -> u32 {
        ((end_frame as i32) - (start_frame as i32) - 1).max(1) as u32
    };

    let velocity = |activation_prob: f32| -> u32 {
        // TODO(knielsen): Implement this
        7
    };
    let decay_function = |activation_prob: f32, t: f32| -> f32 {
        if t < 5.0 {
            activation_prob
        } else {
            activation_prob * (-0.02 * t).exp()
        }
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

            // Handle case where a currently playing note stopped playing
            if let Some((started_at, activation_prob)) = currently_playing[key] {
                if probs[(frame, key)].as_() < deactivation_threshold {
                    // Emit the event and stop playing
                    events.push((started_at as u32, key as u32, duration(frame, started_at), velocity(activation_prob)));
                    currently_playing[key] = None;
                }
            }

            if frame + 1 < num_frames && probs[(frame, key)].as_() < probs[(frame + 1, key)].as_() {
                // We will handle this key in the next frame
                continue
            }

            if probs[(frame, key)].as_() > activation_threshold {
                if let Some((started_at, activation_prob)) = currently_playing[key] {
                    // Either the key is already playing, and we may have a re-activation
                    let time_since_activation = frame as f32 - started_at as f32;
                    if probs[(frame, key)].as_() > decay_function(activation_prob, time_since_activation) {
                        events.push((started_at as u32, key as u32, duration(frame, started_at), velocity(activation_prob))); // Close the old event
                        currently_playing[key] = Some((frame, get_activation_prob()));
                    }
                } else {
                    // Otherwise it is not playing, and we should start playing
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
