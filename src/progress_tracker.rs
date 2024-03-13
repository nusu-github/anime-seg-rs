use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::image_processor::ImageProcessor;
use crate::model::MaskPredictor;

pub(crate) struct ProgressTracker {
    progress_bar: ProgressBar,
    image_paths: Vec<PathBuf>,
}

impl ProgressTracker {
    pub(crate) fn new(input_dir: &Path) -> Self {
        let image_paths: Vec<_> = WalkDir::new(input_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter(|e| image::ImageFormat::from_path(e.path()).is_ok())
            .map(|e| e.into_path())
            .collect();
        let progress_bar = ProgressBar::new(image_paths.len() as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        Self {
            progress_bar,
            image_paths,
        }
    }

    pub(crate) fn process_images(&self, image_processor: &ImageProcessor, model: &MaskPredictor) {
        self.image_paths.par_iter().for_each(|path| {
            match image_processor.process_image(path.as_path(), model) {
                Ok(_) => self.progress_bar.inc(1),
                Err(e) => {
                    if cfg!(debug_assertions) {
                        println!("{:#?}", e);
                    } else {
                        println!("Error processing {:?}", path.display());
                    }
                }
            }
        });
    }
}
