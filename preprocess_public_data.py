from generative_recommenders.research.data.preprocessor import get_common_preprocessors


def main():
    text_embedding_model = "blair"
    #text_embedding_model = "openai" #use OpenAI text-embedding-3-large embeddings instead of BLaIR

    # get_common_preprocessors(text_embedding_model=text_embedding_model)["amzn23_office"].preprocess_rating()
    # get_common_preprocessors(text_embedding_model=text_embedding_model)["amzn23_game"].preprocess_rating()
    get_common_preprocessors(text_embedding_model=text_embedding_model)["amzn23_music"].preprocess_rating()
if __name__ == "__main__":
    main()
