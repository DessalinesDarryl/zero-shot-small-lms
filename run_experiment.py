# run_experiment.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import json
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from collections import Counter
import random


def load_prompt(prompt_file: str):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Nom du dataset (ex: sms)")
    parser.add_argument('--model', type=str, default="MBZUAI/LaMini-Flan-T5-783M", help="Nom du modèle HF")
    parser.add_argument('--prompt', type=str, default=None, help="Chemin vers le fichier de prompt (json)")
    parser.add_argument('--max_samples', type=int, default=100, help="Nb de samples à traiter")
    parser.add_argument('--scoring', choices=['decoding', 'label'], default="decoding", help="Méthode de scoring")
    parser.add_argument('--save_results', action='store_true', help="Sauver les résultats dans /results")
    args = parser.parse_args()

    print(f"Chargement du modèle {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt_file = args.prompt if args.prompt else f"prompts/{args.dataset}.json"
    prompt_info = load_prompt(prompt_file)
    prompt_template = prompt_info["prompt"]
    verbalizer = prompt_info["verbalizer"]
    samples = prompt_info["samples"][:args.max_samples]

    print("Début de l’évaluation...\n")
    y_true, y_pred = [], []

    for i, example in enumerate(tqdm(samples)):
        text = example["text"]
        expected = example["label"]
        y_true.append(expected)

        if args.scoring == "decoding":
            prompt = prompt_template.replace("{TEXT}", text)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=10)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

            pred = None
            for label, synonyms in verbalizer.items():
                if isinstance(synonyms, list):
                    if any(s.lower() in decoded for s in synonyms):
                        pred = label
                        break
                else:
                    if synonyms.lower() in decoded:
                        pred = label
                        break

            y_pred.append(pred if pred else "UNK")

            print(f"\nExemple {i+1}:")
            print(f"Texte     : {text}")
            print(f"Attendu   : {expected}")
            print(f"Prédit    : {pred}")
            print(f"Réponse brute : {decoded}")
            print("-" * 50)

        elif args.scoring == "label":
            scores = {}
            for label, surface in verbalizer.items():
                if isinstance(surface, list):
                    surface = surface[0]  # on prend le 1er synonyme
                prompt = prompt_template.replace("{TEXT}", text).replace("{LABEL}", surface)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                scores[label] = -loss

            pred = max(scores, key=scores.get)
            y_pred.append(pred)

            print(f"\nExemple {i+1}:")
            print(f"Texte     : {text}")
            print(f"Attendu   : {expected}")
            print("Scores    :", {k: round(v, 4) for k, v in scores.items()})
            print(f"Prédit    : {pred}")
            print("-" * 50)

    # Métriques
    filtered = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yp != "UNK"]
    if filtered:
        y_true_f, y_pred_f = zip(*filtered)
        y_true_f = [int(y) for y in y_true_f]
        y_pred_f = [int(y) for y in y_pred_f]
        acc = accuracy_score(y_true_f, y_pred_f)

        if len(set(y_true_f)) > 2:
            f1 = f1_score(y_true_f, y_pred_f, average="weighted")
        else:
            f1 = f1_score(y_true_f, y_pred_f, average="binary")

        # Baselines
        majority_class = Counter(y_true_f).most_common(1)[0][0]
        y_majority = [majority_class] * len(y_true_f)
        acc_majority = accuracy_score(y_true_f, y_majority)
        f1_majority = f1_score(y_true_f, y_majority, average="weighted")

        random.seed(42)
        labels = list(set(y_true_f))
        y_random = [random.choice(labels) for _ in y_true_f]
        acc_random = accuracy_score(y_true_f, y_random)
        f1_random = f1_score(y_true_f, y_random, average="weighted")

        print("\n=== Résultats ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("\n=== Baselines ===")
        print(f"Majority - Accuracy: {acc_majority:.4f} | F1: {f1_majority:.4f}")
        print(f"Random   - Accuracy: {acc_random:.4f} | F1: {f1_random:.4f}")

    else:
        print("\nAucune prédiction exploitable.")
        acc, f1 = None, None

    if args.save_results:
        Path("results").mkdir(exist_ok=True)
        prompt_name = Path(prompt_file).stem
        out_path = f"results/{prompt_name}_{args.scoring}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "dataset": args.dataset,
                "scoring": args.scoring,
                "y_true": y_true,
                "y_pred": y_pred,
                "accuracy": acc,
                "f1": f1,
                "baseline_majority": {"accuracy": acc_majority, "f1": f1_majority},
                "baseline_random": {"accuracy": acc_random, "f1": f1_random}
            }, f, indent=2)

        print(f"\nRésultats sauvegardés dans {out_path}")

if __name__ == "__main__":
    main()
