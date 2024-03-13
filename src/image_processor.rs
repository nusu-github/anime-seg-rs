use std::path::Path;

use anyhow::Result;

use crate::Config;
use crate::model::MaskPredictor;

pub(crate) struct ImageProcessor {
    config: Config,
}

impl ImageProcessor {
    pub(crate) fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub(crate) fn process_image(&self, path: &Path, model: &MaskPredictor) -> Result<()> {
        let input_dir = Path::new(&self.config.input_dir);
        let output_dir_base = Path::new(&self.config.output_dir);
        let output_format = &self.config.format;

        let relative_path = path.strip_prefix(input_dir)?;
        let output_dir = output_dir_base.join(relative_path.parent().unwrap_or(Path::new("")));

        std::fs::create_dir_all(&output_dir)?;

        let masked_image = model.predict(path)?;
        let filename = path.file_stem().unwrap().to_str().unwrap();
        let output_file = output_dir.join(format!("{}.{}", filename, output_format));

        masked_image.save(output_file)?;

        Ok(())
    }
}
