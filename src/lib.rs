pub use std::fs;
pub use std::collections::HashMap;
pub use std::path::Path;
pub use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct TrainedModel {
    pub vocabulary: Vec<String>,
    pub mine_probs: HashMap<String, f64>,
    pub others_probs: HashMap<String, f64>,
    pub mine_prior: f64,
    pub others_prior: f64,
}

pub fn save_model(model: &TrainedModel, path: impl AsRef<Path>) {
    let json = serde_json::to_string_pretty(model)
        .expect("Falha ao serializar modelo");
    fs::write(path, json)
        .expect("Falha ao escrever arquivo");
}

pub fn load_model(path: impl AsRef<Path>) -> TrainedModel {
    let json = fs::read_to_string(path)
        .expect("Falha ao ler arquivo do modelo");
    serde_json::from_str(&json)
        .expect("Falha ao deserializar modelo")
}

pub fn word_probabilities(
    class_counts: &HashMap<String, u32>,
    vocabulary: &[String],
) -> HashMap<String, f64> {
    let total: u32 = vocabulary
        .iter()
        .map(|word| class_counts.get(word).unwrap_or(&0))
        .sum();

    let vocab_size = vocabulary.len() as u32;

    vocabulary
        .iter()
        .map(|word| {
            let count = *class_counts.get(word).unwrap_or(&0);
            let prob = (count + 1) as f64 / (total + vocab_size) as f64;
            (word.clone(), prob)
        })
        .collect()
}

pub fn process_directory(directory: impl AsRef<Path>) -> Vec<HashMap<String, u32>> {
    let mut all_counts: Vec<HashMap<String, u32>> = Vec::new();

    for path in list_files(directory){
        let counts = process_file(&path);
        all_counts.push(counts);
    }
    all_counts
}

pub fn process_file(path: impl AsRef<Path>) -> HashMap<String, u32> {
    let content = match fs::read_to_string(&path){
        Err(why) => panic!("couldn't open because: {}", why),
        Ok(file) => file,
    };
    let content_without_frontmatter = remove_frontmatter(&content).to_string();
    let content_without_metadata = remove_export_metadata(&content_without_frontmatter);

    let cleaned: String = content_without_metadata
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();
    let words: Vec<&str> = cleaned.split_whitespace().collect();

    let counts = count_words(&words);
    counts
}

pub fn remove_frontmatter(content: &str) -> &str {
    if content.starts_with("---") {
        if let Some(end) = content[3..].find("---") {
            let after = 3 + end + 3; //pula os dois marcadores e o conteúdo entre eles
            return content[after..].trim_start();
        }
    }
    content
}

pub fn remove_export_metadata(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();

    let mut metadata_indices = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim_start();

        if trimmed.starts_with("![") {
            metadata_indices.push(i);
            break;
        }

        if !(trimmed.starts_with("#") || trimmed.starts_with(">") || trimmed.starts_with("*") || trimmed.starts_with("![")) {
            break;
        }

        if trimmed.starts_with("*") {
            metadata_indices.push(i);
        }   
        i += 1;
    }

    lines
        .iter()
        .enumerate()
        .filter(|(i, _)| !metadata_indices.contains(i))
        .map(|(_, line)| *line)
        .collect::<Vec<&str>>()
        .join("\n")
}

pub fn count_words(words: &[&str]) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = HashMap::new();

    for word in words {
        let counter = counts.entry(word.to_string()).or_insert(0);
        *counter += 1;
    }
    counts
}

pub fn print_sorted_count(counts: &HashMap<String, u32>) {
    let mut sorted: Vec<(&String, &u32)> = counts.iter().collect();

    sorted.sort_by(|a, b| a.1.cmp(b.1));

    for (word, count) in &sorted {
        println!("{}: {}", word, count);
    }
}

pub fn top_n_words(counts: &HashMap<String, u32>, n: usize) -> Vec<String> {
    let mut sorted: Vec<(&String, &u32)> = counts.iter().collect();
    sorted.sort_by_key(|(_, count)| std::cmp::Reverse(**count));

    sorted
        .into_iter()
        .take(n)
        .map(|(word, _)| word.clone())
        .collect()
}

pub fn count_files(dir: impl AsRef<Path>) -> usize {
    fs::read_dir(dir)
        .expect("Falha ao ler diretório")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .count()
}

pub fn list_files(dir: impl AsRef<Path>) -> Vec<std::path::PathBuf> {
    fs::read_dir(dir)
        .expect("Falha ao ler diretório")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect()
}

pub fn sum_counts(counts: &Vec<HashMap<String, u32>>) -> HashMap<String, u32> {
    let mut result: HashMap<String, u32> = HashMap::new();
    for count in counts {
        for word in count {
            let counter = result
            .entry(
                (*word.0
                .clone())
                .to_string()
            )
            .or_insert(0);
            *counter += word.1;
        }
    }

    result
}

pub fn filter_vocabulary(count: HashMap<String, u32>, vocabulary: Vec<String>) -> HashMap<String, u32> {
    let filtered_text: HashMap<String, u32> = count.into_iter()
    .filter(|(word, _)| vocabulary.contains(&word))
    .collect();

    filtered_text
}