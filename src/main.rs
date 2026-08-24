use classifier::*;

fn main() {
    let model = load_model("/mnt/windows/Junior/Projects/Classifier/src/model.json");
    let text_count = process_file("/mnt/windows/Junior/Projects/Classifier/data/test/others/episodio-1-por-que-voce-nao-fica.md");
    
    let filtered_text = filter_vocabulary(text_count, model.vocabulary);
    //print_sorted_count(&filtered_text);

    let mut mine_proximity: f64 = model.mine_prior.ln();

    for (word, number) in filtered_text.clone() {
        let mine_prob = model.mine_probs.get(&word).unwrap();
        mine_proximity += (number as f64) * mine_prob.ln();
    }
    println!("Mine proximity: {}", mine_proximity);

    let mut others_proximity: f64 = model.others_prior.ln();

    for (word, number) in filtered_text {
        let others_prob = model.others_probs.get(&word).unwrap();
        others_proximity += (number as f64) * others_prob.ln();
    }
    println!("Others proximity: {}", others_proximity);

    if mine_proximity > others_proximity{
    } else {
    }
}