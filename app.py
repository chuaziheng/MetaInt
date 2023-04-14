import streamlit as st
import pandas as pd
from data import INPUT_EXAMPLES_DICT, WORD_2_CLS_DICT
from inference import conduct_metaphor_interpretation_task1, conduct_metaphor_interpretation_task2
from model import load_model_1, load_model_2, load_tokenizer

@st.cache_resource
def get_models():
    model1 = load_model_1()
    model2 = load_model_2()
    tokenizer = load_tokenizer()
    st.success("Loaded models")
    return model1, model2, tokenizer

def main():
    def process_output_task2(info):
        display1 = ""
        for idx, i in enumerate(info):
            sentence = f"Sentence {idx + 1}: {i}\n"
            display1 += sentence
        return display1



    st.set_page_config(layout="wide")  # Set the page layout to wide
    st.markdown("<h1 style='text-align: center;'>Metaphor Interpretation Demo</h1>", unsafe_allow_html=True)

    # Add a container to split the screen into two vertical halves

    left_col, right_col = st.columns(2)

    # Add a large text box to the left half of the screen
    with left_col:
        # task checkbox
        col1_checkbox, col2_checkbox = st.columns(2)
        with col1_checkbox:
            select_task1 = st.checkbox('Task 1 (CLS)', value=True)
        with col2_checkbox:
            select_task2 = st.checkbox('Task 2 (MLM)', value=True)
        # Add a dropdown below the title
        option = st.selectbox("Select an option:", INPUT_EXAMPLES_DICT.keys())

        input_metaphor_sentence = st.text_area("Enter your text here:", height=200, value=INPUT_EXAMPLES_DICT[option]['metaphor_sentence'])

        col1, col2 = st.columns(2)
        LABEL_CHOICES = list(WORD_2_CLS_DICT.keys())
        if select_task1:
            with col1:
                input_positive = st.selectbox("Positive", LABEL_CHOICES, index=LABEL_CHOICES.index(INPUT_EXAMPLES_DICT[option]['positive']))
                input_neg_2 = st.selectbox("Negative 2", LABEL_CHOICES, index=LABEL_CHOICES.index(INPUT_EXAMPLES_DICT[option]['neg_2']))
            with col2:
                input_neg_1 = st.selectbox("Negative 1", LABEL_CHOICES, index=LABEL_CHOICES.index(INPUT_EXAMPLES_DICT[option]['neg_1']))
                input_neg_3 = st.selectbox("Negative 3", LABEL_CHOICES, index=LABEL_CHOICES.index(INPUT_EXAMPLES_DICT[option]['neg_3']))
        else:
            with col1:
                input_positive = st.text_input("Positive", value=INPUT_EXAMPLES_DICT[option]['positive'])
                input_neg_2 = st.text_input("Negative 2", value=INPUT_EXAMPLES_DICT[option]['neg_2'])
            with col2:
                input_neg_1 = st.text_input("Negative 1", value=INPUT_EXAMPLES_DICT[option]['neg_1'])
                input_neg_3 = st.text_input("Negative 3", value=INPUT_EXAMPLES_DICT[option]['neg_3'])

        # start loading first
        model_1, model_2, tokenizer = get_models()
        # Add a submit button
        if st.button("Submit"):
            input_dict = {
                'input_sentence_raw': input_metaphor_sentence,
                'pos_label': input_positive,
                'neg_label_1': input_neg_1,
                'neg_label_2': input_neg_2,
                'neg_label_3': input_neg_3,
            }
            with st.spinner(text="Inferencing..."):
                res1 = [None]*4
                res2 = [None]*4
                info2 = ""
                if select_task1:
                    res1 = conduct_metaphor_interpretation_task1(model_1, tokenizer, input_data=input_dict)
                    res1 = [f"{i:.2f}" for i in res1]
                if select_task2:
                    res2, info2 = conduct_metaphor_interpretation_task2(model_2, tokenizer, input_data=input_dict)
                    res2 = [f"{i:.2f}" for i in res2]


            # Add text to the right half of the screen
            with right_col:
                st.write('Results:')
                df = pd.DataFrame({
                    'task 1': res1,
                    'task 2': res2,
                }, index=[('POS', input_positive), ('NEG 1', input_neg_1), ("NEG 2", input_neg_2), ("NEG 3", input_neg_3)])

                st.dataframe(df.style.highlight_max(axis=0))
                if select_task2:
                    st.text_area(label="Task 2 Processing:", value=process_output_task2(info2), height=250)




if __name__ == "__main__":
    main()
