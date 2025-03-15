import random
import string
import os

def ruin_sentence(sentence):
    """
    Takes an input sentence and applies random noise for words and characters.
        Swap two adjacent characters in a word;
        Delete a random character;
        Substitute a character with a random letter;
        Occasionally swap adjacent words.
    """
    words = sentence.split()
    corrupted_words = []
    
    for word in words:
        # 30% probability of applying a corruption to the word
        if random.random() < 0.3:
            method = random.choice(['swap', 'delete', 'substitute'])
            if len(word) > 1:
                if method == 'swap':  # Swap two adjacent characters at a random index
                    i = random.randint(0, len(word) - 2)
                    word = word[:i] + word[i+1] + word[i] + word[i+2:]
                elif method == 'delete':  # Delete a character randomly
                    i = random.randint(0, len(word) - 1)
                    word = word[:i] + word[i+1:]
                elif method == 'substitute':  # Substitute a character with a random letter
                    i = random.randint(0, len(word) - 1)
                    new_char = random.choice(string.ascii_lowercase)
                    word = word[:i] + new_char + word[i+1:]
        corrupted_words.append(word)
    
    # 20% probability of swapping two adjacent words in the sentence
    if len(corrupted_words) > 1 and random.random() < 0.2:
        i = random.randint(0, len(corrupted_words) - 2)
        corrupted_words[i], corrupted_words[i+1] = corrupted_words[i+1], corrupted_words[i]
    
    corrupted_sentence = " ".join(corrupted_words)
    
    # Make sure corrupted sentence is different from the original input
    if corrupted_sentence == sentence:
        if corrupted_words:
            first_word = corrupted_words[0]
            if len(first_word) > 0:
                new_char = random.choice(string.ascii_lowercase)
                corrupted_words[0] = new_char + first_word[1:]
        corrupted_sentence = " ".join(corrupted_words)
    
    return corrupted_sentence

def generate_ruined_dataset(input_file, output_file):
    """
    Reads the training data from input_file, extracts the natural sentences in the first first column,
    applies the ruin_sentence function to create a corruption, and writes the results to output_file.
    
    Each output line is formatted as "natural_sentence <TAB> corrupted_sentence"
    """
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split('\t')
            if not parts:
                continue
            
            original = parts[0].strip()
            if not original:
                continue
            corrupted = ruin_sentence(original)
            
            if corrupted == original:
                continue
            f_out.write(f"{original}\t{corrupted}\n")

def main():
    input_file = "challenge-data/train.txt"
    solution_dir = "solution"
    os.makedirs(solution_dir, exist_ok=True)
    output_file = os.path.join(solution_dir, "part2.txt")
    generate_ruined_dataset(input_file, output_file)
    print(f"Corrupted dataset saved to {output_file}")

if __name__ == "__main__":
    main()