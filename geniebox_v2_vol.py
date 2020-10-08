import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns                   
import matplotlib.pyplot as plt             
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Format website
page_bg_img = '''
<style>
body {
background-image: url("https://geniebox-media.s3-us-west-1.amazonaws.com/geniebox_background.png");
background-size: 400px;
background-repeat: no-repeat;
background-position: right 5% bottom 300%;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title('GenieBox - Predicting The Most Suitable Shipping Box Size')
st.header('Machine Learning Powered Genie®') 


# Define model as a function

@st.cache 
def build_model():
	df = pd.read_csv('/Users/elle.le/Workspace/BoxSize_local/new_data/GM_box_size_clean_data.csv')
	# Keep the ratio of different box sizes
	x_train,x_test,y_train, y_test = train_test_split(df.iloc[:,0:5], df['FedEx_box_type'], stratify = df['FedEx_box_type'])

	dt = DecisionTreeClassifier(class_weight='balanced') 
	dt.fit(x_train,y_train)
	print(f'Test Score of the Model {dt.score(x_test,y_test)}')
	return dt

model = build_model()

df2=pd.read_csv('/Users/elle.le/Workspace/BoxSize_local/new_data/Geniebox_menu.csv')

options = st.sidebar.multiselect(
	'What items do you want to buy?',
	df2['variant_id_name'])
st.write('You selected:')

for option in options:
	st.image(df2['image_url'].loc[df2['variant_id_name'] == option].item(), width=222,
		caption=option)

input_raw = np.array(options)
#st.write('first input: ', input_raw[0])

input_coded =[]
#for i in input_raw:
#	input_coded.append(df2.loc[df2['variant_id_name'] == i,'item_label'])
total_vol = 0
for item in input_raw:
	val = df2.loc[df2['variant_id_name'] == item, 'item_label'].values[0]
	total_vol += df2.loc[df2['variant_id_name'] == item, 'item_vol'].values[0]
	input_coded.append(val)

count_item_label = pd.DataFrame(np.zeros(4).reshape(4,1), columns = ['Count'], 
				index = ['item_size_0', 'item_size_1', 'item_size_2', 'item_size_3'])
index, counts = np.unique(input_coded, return_counts=True)

# Model Input: count_item_label, needed to reshape(1,-1) 
count_item_label.iloc[index,0] = counts

print(count_item_label)
model_input = np.zeros(5)
model_input[-1]=total_vol
model_input[:4]=count_item_label.iloc[:,0]
print(model_input)

box_id_map = {0:'Small', 1:'Medium', 2:'Large'}
st.sidebar.subheader('The best box size for your order is: ')

# Fit into the model 
st.sidebar.write(box_id_map[model.predict(model_input.reshape(1,-1))[0]])

#st.markdown("![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")
#st.markdown("![Alt Text](https://dy6g3i6a1660s.cloudfront.net/d9Ej2RbHy4jWMyJi2IOAyQk6CTo/zb_p.jpg)")
#st.image("https://dy6g3i6a1660s.cloudfront.net/d9Ej2RbHy4jWMyJi2IOAyQk6CTo/zb_p.jpg", width=333)

#sdhfdhsfksdhf xxxxxhfjsdhfffejjhfsjjhif
