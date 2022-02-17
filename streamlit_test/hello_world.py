import streamlit as st
st.write('Hello world')
import matplotlib.pyplot as plt
fig = plt.figure()
plt.scatter([1,2,3],[1,4,9])
st.write(fig)

# Put things in sidebar
file=st.sidebar.file_uploader('load file')
st.sidebar.write(file.name)
