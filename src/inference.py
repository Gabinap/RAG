from transformers import GenerationConfig, pipeline


def infer():
    generator = pipeline(
        task="text-generation",
        model="Qwen/Qwen3-0.6B",
        clean_up_tokenization_spaces=False,
    )

    config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=200,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    # print in red
    print(f"\033[91m{config}\033[0m")
    response = generator(
        "the secret to baking a really good cake is ",
        generation_config=config,
    )
    # print in blue
    print(f"\033[94m{response[0].get('generated_text')}\033[0m")