def calculate_accuracy(file1_path: str, file2_path: str) -> float:
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines1 = [line.strip() for line in file1.readlines()]
        lines2 = [line.strip() for line in file2.readlines()]
            
        total_lines = len(lines1)
        if total_lines == 0:
            return 0
            
        matched_lines = sum(1 for i in range(total_lines) if lines1[i] == lines2[i])
            
        accuracy = (matched_lines / total_lines) * 100
        return accuracy

if __name__ == "__main__":
    file1_path = "test\part1_answer.txt"  # Small portion of solution of test.rand.txt labeled by human
    file2_path = "solution\part1.txt"  # Answer geneated by is_it_english.py
    accuracy = calculate_accuracy(file1_path, file2_path)
    print(f"Accuracy: {accuracy:.2f}%")