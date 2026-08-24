use classifier::*;

fn main() {    
    let my_train = Path::new("/mnt/windows/Junior/Projects/Classifier/data/train/mine");
    let others_train = Path::new("/mnt/windows/Junior/Projects/Classifier/data/train/others");

    let mine_counts = process_directory(my_train);
    let others_counts = process_directory(others_train);
    
    let mine_sum = sum_counts(&mine_counts);
    let others_sum = sum_counts(&others_counts);

    let total_vec: Vec<HashMap<String, u32>> = vec![mine_sum.clone(), others_sum.clone()];
    let total_sum = sum_counts(&total_vec);
    
    let vocabulary = top_n_words(&total_sum, 100);

    let mine_probs = word_probabilities(&mine_sum, &vocabulary);
    let others_probs = word_probabilities(&others_sum, &vocabulary);

    let my_train_size = count_files(my_train);
    let others_train_size = count_files(others_train);
    let total_size = my_train_size + others_train_size;
    
    let priors = vec![my_train_size as f64 / total_size as f64, others_train_size as f64 / total_size as f64];

    let model = TrainedModel {
        vocabulary,
        mine_probs,
        others_probs,
        mine_prior: priors[0],
        others_prior: priors[1],
    };

    save_model(&model, "model.json");
}