
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
    _capacity: usize,
}

#[no_mangle]
pub extern "C" fn extract_midi_events(batch_size: i32, num_frames: i32, frame_size: i32, data: *const f32) -> *mut MidiEventList {
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
    let length = result.len();
    let _capacity = result.capacity();
    let ptr = result.as_ptr() as *mut MidiEvent;
    std::mem::forget(result);
    Box::into_raw(Box::new(MidiEventList { ptr, length, _capacity }))
}

#[no_mangle]
pub extern "C" fn free_midi_events(ptr: *mut MidiEventList) {
    if !ptr.is_null() {
        unsafe {
            let vec_ptr = (*ptr).ptr;
            let vec_length = (*ptr).length;
            let vec_capacity = (*ptr)._capacity;
            Vec::from_raw_parts(vec_ptr, vec_length, vec_capacity);
            Box::from_raw(ptr);
        }
    }
}
