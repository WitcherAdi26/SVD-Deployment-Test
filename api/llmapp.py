from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def returnchar():
    q = request.args.get('query')
    import langchain
    from langchain import FewShotPromptTemplate, PromptTemplate, HuggingFaceHub
    from langchain.memory import ConversationBufferMemory
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    import pandas as pd
    # import getpass
    import os

    # inference_api_key = getpass.getpass("Enter your HF Inference API Key:")
    os.environ['HUGGINGFACEHUB_API_TOKEN']="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"
    llm = HuggingFaceHub(repo_id ='mistralai/Mistral-7B-Instruct-v0.1')
    data=pd.read_csv("D:\\IIITP\\Flipkart 5.0\\Data\\api\\Dataset2 - Copy.csv",header=0)

    subcategory_counts = data['product_sub_catagory'].value_counts()
    print(len(subcategory_counts))
    sorted_subcategories = subcategory_counts.index

    top_subcategories = list(sorted_subcategories[:1900])
    top_subcategories=",".join(top_subcategories)

    del data,subcategory_counts, sorted_subcategories

    examples = [
        {
            "question":"What are some products I should buy if I'm already buying shoes",
            "answer":'''
            Socks,Shoe Polish,Insoles
            '''
        },
        {
            "question":"What are some products I should buy if I'm already buying laptops",
            "answer":'''
            Mouse,Laptop Case,Keyboard
            '''
        },
        {
            "question":"What are some products I should buy if I'm already buying school bags",
            "answer":'''
            Books,Pens,Pencils
            '''
        },
        {
            "question":"What are some products I should buy if I'm already buying beds",
            "answer":'''
            +
            Blankets,Pillows,Bedsheet
            '''
        },
        {
            "question":"What are some products I should buy if I'm already buying toys",
            "answer":'''
            Chocolates,Arts and Craft,Board Games
            '''
        }
    ]

    example_template = """
    User: {question}
    AI: {answer}
    """
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template=example_template
    )

    prefix = """ The assisstant is a world class recommender of products. The assisstant gives 3 most appropriate items most compatible with product in question
    Here are some examples of excerpts between user and the assiatant. Give only 3 products as the output from: """+top_subcategories

    suffix = """
    User: {question}
    AI: """

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["question"],
        example_separator="\n\n"
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    chain = LLMChain(llm=llm, prompt=few_shot_prompt_template,verbose=True,memory=memory)
    x=chain.predict(question="What are 3 unique products I should buy with "+q).split(',')
    print(x)
    res=[]
    res.append(x[0][13::])
    res.append(x[1])
    word=[]
    for i in range(len(x[2])):
        if(x[2][i]=='\n'):
            word.append(x[2][:i])
            break
    res.append(word[0])
    return {"recommendations":res}

if __name__ == "__main__":
    app.run(port=3000)
