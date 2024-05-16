
#[repr(C)]
pub struct MidiEvent {
    attack_time: u8,
    note: u8,
    duration: u8,
    velocity: u8,
}

#[repr(C)]
pub struct MidiEventList {
    ptr: *mut MidiEvent,
    length: usize,
}

#[no_mangle]
pub extern "C" fn extract_midi_events(batch_size: int, num_frames: int, frame_size: int, data: *const f32) -> *mut MidiEventList {
    let mut result = vec![];
    result.push(MidiEvent {
        attack_time: 0,
        note: 0,
        duration: 0,
        velocity: 0,
    });
    result.push(MidiEvent {
        attack_time: 1,
        note: 2,
        duration: 3,
        velocity: 4,
    });

    // The caller will be responsible for calling `free_midi_events` on the returned list
    let len = result.len();
    let ptr = result.as_ptr() as *mut MidiEvent;
    std::mem::forget(result);
    Box::into_raw(Box::new(MidiEventList {
        ptr: ptr,
        length: len
    }))
}

#[no_mangle]
pub extern "C" fn free_midi_events(ptr: *mut MidiEventList) {
    if !ptr.is_null() {
        unsafe {
            let vec_ptr = (*ptr).ptr;
            Vec::from_raw(vec_ptr);
            Box::from_raw(ptr);
        }
    }
}
