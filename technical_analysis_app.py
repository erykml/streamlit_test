import streamlit as st
import pandas as pd
import numpy as np

st.title("Hello")

df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), 
                  columns=list('ABCD'))

st.dataframe(df)