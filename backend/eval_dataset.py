"""
Evaluation test set.

Hand-built ground-truth question/answer pairs based on the actual
content of the test "Evolution of Man" PDF used throughout development.

Used by evaluate_rag.py to measure retrieval and generation quality
(Point E in the study guide) — context precision, context recall,
faithfulness, and answer relevancy.
"""

EVAL_QUESTIONS = [
    {
        "question": "What is phyletic gradualism?",
        "ground_truth": "Phyletic gradualism is a mode of evolution where a species gradually transforms into another species over a continuous series of changes.",
    },
    {
        "question": "What is Australopithecus known for?",
        "ground_truth": "Australopithecus is known for the famous fossil of a female ape-man called Lucy, and represents an early hominin genus with ape-like and human-like traits.",
    },
    {
        "question": "What is Homo erectus also known as?",
        "ground_truth": "Homo erectus is known as upright walking man, and includes Java man and Peking man fossils.",
    },
    {
        "question": "Where were Neanderthal fossils first discovered?",
        "ground_truth": "Neanderthal fossils were first discovered in 1856 in a limestone quarry.",
    },
    {
        "question": "What is the cranial capacity of Homo sapiens mentioned in the document?",
        "ground_truth": "Homo sapiens (modern man) is noted to have a cranial capacity of 1350 cc.",
    },
    {
        "question": "What is Ramapithecus?",
        "ground_truth": "Ramapithecus refers to ancestral ape forms that lived approximately 8 to 14 million years ago.",
    },
    {
        "question": "What is Homo habilis also known as?",
        "ground_truth": "Homo habilis is also known as the 'handy man'.",
    },
    {
        "question": "What is Homo naledi?",
        "ground_truth": "Homo naledi is an extinct species of hominin discovered by anthropologists.",
    },
    {
        "question": "When did Cro-Magnon man's fossil remains get unearthed?",
        "ground_truth": "Cro-Magnon man (Homo sapiens sapiens) is described with its earliest fossil remains being unearthed, representing an early modern human form.",
    },
    {
        # Deliberately NOT covered by the document — tests honest fallback behavior
        "question": "What is the current population of India?",
        "ground_truth": "This information is not present in the uploaded document.",
    },
]