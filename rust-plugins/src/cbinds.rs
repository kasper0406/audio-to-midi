use half::f16;
use ndarray::ArrayView;
use ndarray::ArrayView2;
use ndarray::ArrayView3;
use ndarray::ShapeBuilder;
use ndarray::Ix;
use ndarray::Dim;

#[repr(C)]
pub struct MidiEvent {
    attack_time: u64,
    note: u8,
    duration: u64,
    velocity: u8,
}

#[repr(C)]
pub struct MidiEventList {
    ptr: *mut MidiEvent,
    length: usize,
    _capacity: usize,
}

#[repr(C)]
pub struct MLMultiArrayWrapper<const N: usize> {
    strides: [u64; N],
    dims: [u64; N],
    data: *const u8,
}

macro_rules! define_mlmultiarray_helpers {
    ($($N:expr),*) => {
        $(
            impl MLMultiArrayWrapper<$N> {
                pub fn view<'a, T>(&self) -> ArrayView<'a, T, Dim<[Ix; $N]>> {
                    let dims: [usize; $N] = self.dims.map(|d| d as usize);
                    let strides: [usize; $N] = self.strides.map(|s| s as usize);
                    let shape = Dim(dims).strides(Dim(strides));
                    unsafe { ArrayView::from_shape_ptr(shape, self.data as *const T) }
                }
            }
        )*
    };
}

pub type MLMultiArrayWrapper1 = MLMultiArrayWrapper<1>;
pub type MLMultiArrayWrapper2 = MLMultiArrayWrapper<2>;
pub type MLMultiArrayWrapper3 = MLMultiArrayWrapper<3>;
define_mlmultiarray_helpers!(1, 2, 3);

#[no_mangle]
pub extern "C" fn extract_midi_events(data: MLMultiArrayWrapper3, overlap: f64, duration_per_frame: f64) -> *mut MidiEventList {
    // Actually the pointer is to a f16 array, but we need to expose the pointer as something cbindgen knows about!
    let array_view = data.view::<f16>();

    let stitched = crate::common::stitch_probs(&array_view, overlap, duration_per_frame);
    let raw_events = crate::common::extract_events(&stitched.view());

    let mut events = vec![];
    for (attack_time, note, duration, velocity) in raw_events {
        let (attack_time, note, duration, velocity) = (
            attack_time as u64,
            note as u8,
            duration as u64,
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
