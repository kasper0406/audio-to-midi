use half::f16;
use ndarray::ArrayView;
use ndarray::ShapeBuilder;

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
pub extern "C" fn extract_midi_events(num_frames: i32, frame_stride: i32, num_notes: i32, note_stride: i32, data: *const u8) -> *mut MidiEventList {
    // Actually the pointer is to a f16 array, but we need to expose the pointer as something cbindgen knows about!
    let data = data as *const f16;

    let shape = (num_frames as usize, num_notes as usize).strides((frame_stride as usize, note_stride as usize));
    let array_view = unsafe { ArrayView::from_shape_ptr(shape, data as *const f16) };
    let raw_events = crate::common::extract_events(&array_view);

    let mut events = vec![];
    for (attack_time, note, duration, velocity) in raw_events {
        let (attack_time, note, duration, velocity) = (
            attack_time as u8,
            note as u8,
            duration as u8,
            velocity as u8
        );
        events.push(MidiEvent {
            attack_time, note, duration, velocity
        })
    }

    // The caller will be responsible for calling `free_midi_events` on the returned list
    let length = events.len();
    let _capacity = events.capacity();
    let ptr = events.as_ptr() as *mut MidiEvent;
    std::mem::forget(events);
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
