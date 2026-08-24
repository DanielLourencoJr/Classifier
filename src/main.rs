use classifier::*;

fn main() {
    let model = load_model("/mnt/windows/Junior/Projects/Classifier/src/model.json");
    test(model);
}