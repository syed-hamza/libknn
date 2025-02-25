// Function to create a confusion matrix
std::vector<std::vector<int>> compute_confusion_matrix(
    const std::vector<float>& predictions,
    const std::vector<float>& truths,
    std::unordered_map<float, int>& label_to_index,
    std::vector<int>& support
) {
    std::unordered_set<float> unique_labels(truths.begin(), truths.end());
    int num_classes = unique_labels.size();

    // Map labels to indices
    int index = 0;
    for (const auto& label : unique_labels) {
        label_to_index[label] = index++;
    }

    // Initialize confusion matrix
    std::vector<std::vector<int>> confusion_matrix(num_classes, std::vector<int>(num_classes, 0));
    support.resize(num_classes, 0);

    // Fill the confusion matrix
    for (size_t i = 0; i < truths.size(); i++) {
        int actual = label_to_index[truths[i]];
        int predicted = label_to_index[predictions[i]];
        confusion_matrix[actual][predicted]++;
        support[actual]++;
    }

    return confusion_matrix;
}

// Function to compute precision for each class
std::vector<float> compute_precision(const std::vector<std::vector<int>>& confusion_matrix) {
    int num_classes = confusion_matrix.size();
    std::vector<float> precision(num_classes, 0.0f);

    for (int i = 0; i < num_classes; i++) {
        int TP = confusion_matrix[i][i];
        int FP = 0;
        for (int j = 0; j < num_classes; j++) {
            if (i != j) {
                FP += confusion_matrix[j][i];  // False Positives
            }
        }
        precision[i] = (TP + FP) ? static_cast<float>(TP) / (TP + FP) : 0.0f;
    }

    return precision;
}

// Function to compute recall for each class
std::vector<float> compute_recall(const std::vector<std::vector<int>>& confusion_matrix) {
    int num_classes = confusion_matrix.size();
    std::vector<float> recall(num_classes, 0.0f);

    for (int i = 0; i < num_classes; i++) {
        int TP = confusion_matrix[i][i];
        int FN = 0;
        for (int j = 0; j < num_classes; j++) {
            if (i != j) {
                FN += confusion_matrix[i][j];  // False Negatives
            }
        }
        recall[i] = (TP + FN) ? static_cast<float>(TP) / (TP + FN) : 0.0f;
    }

    return recall;
}

// Function to compute F1-score for each class
std::vector<float> compute_f1_score(const std::vector<float>& precision, const std::vector<float>& recall) {
    int num_classes = precision.size();
    std::vector<float> f1_score(num_classes, 0.0f);

    for (int i = 0; i < num_classes; i++) {
        f1_score[i] = (precision[i] + recall[i]) ? 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) : 0.0f;
    }

    return f1_score;
}

// Function to generate the classification report
void classification_report(const std::vector<float>& predictions, const std::vector<float>& truths) {
    if (predictions.size() != truths.size()) {
        throw std::invalid_argument("Size mismatch between predictions and ground truths");
    }

    std::unordered_map<float, int> label_to_index;
    std::vector<int> support;
    
    // Compute confusion matrix
    std::vector<std::vector<int>> confusion_matrix = compute_confusion_matrix(predictions, truths, label_to_index, support);

    // Compute precision, recall, and F1-score
    std::vector<float> precision = compute_precision(confusion_matrix);
    std::vector<float> recall = compute_recall(confusion_matrix);
    std::vector<float> f1_score = compute_f1_score(precision, recall);

    // Compute macro and weighted averages
    float macro_precision = 0.0f, macro_recall = 0.0f, macro_f1 = 0.0f;
    float weighted_precision = 0.0f, weighted_recall = 0.0f, weighted_f1 = 0.0f;
    int total_samples = 0;

    for (size_t i = 0; i < precision.size(); i++) {
        macro_precision += precision[i];
        macro_recall += recall[i];
        macro_f1 += f1_score[i];

        weighted_precision += precision[i] * support[i];
        weighted_recall += recall[i] * support[i];
        weighted_f1 += f1_score[i] * support[i];

        total_samples += support[i];
    }

    int num_classes = precision.size();
    macro_precision /= num_classes;
    macro_recall /= num_classes;
    macro_f1 /= num_classes;

    if (total_samples > 0) {
        weighted_precision /= total_samples;
        weighted_recall /= total_samples;
        weighted_f1 /= total_samples;
    }

    // Print classification report
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nClassification Report:\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Class | Precision | Recall | F1-score\n";
    std::cout << "--------------------------------------\n";

    for (const auto& pair : label_to_index) {
        int i = pair.second;
        std::cout << pair.first << "     | " << precision[i] << "     | " << recall[i] << "   | " << f1_score[i] << "\n";
    }

    std::cout << "--------------------------------------\n";
    std::cout << "Macro Avg    | " << macro_precision << "   | " << macro_recall << "   | " << macro_f1 << "\n";
    std::cout << "Weighted Avg | " << weighted_precision << "   | " << weighted_recall << "   | " << weighted_f1 << "\n";
}