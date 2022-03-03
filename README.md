# Predict_Care
#### Machine Learning techniques in healthcare use the increasing amount of health data to improve patient outcome. Many of these techniques focus on diagnosis and prediction. 
In this project I decided to build a website to check if a patient has any cardiovascular disease based on the input parameters and also it detects if the tumor is malignant or benign for a patient with breast cancer. 
> I have designed a UI that is a webpage connected to the python file where predictions will be made and the inputs from the UI reach python file through flask. After the prediction is made the output is sent through flask a a final result page is displayed with the outcome and accuracy of prediction. I tried to calculate the correlation among the attributes in order to have a better understanding of the important attributes with respect to classification as well as the association among the values. After correlation checking and splitting of dataset for training and testing I used different Machine learning algorithms such as Naive Bayes, Logistic Regression, Random Forest and Decision Tree. 
> And for each of these algorithms I calculated the accuracy, precision and recall values from the Confusion matrix in order to see which of the attributes perform better with the test data and whichever algorithm performed better was chosen for the final classification to be made from the inputs coming from webpage.
> It was seen that for Breast cancer classification Random Forest was performing best and for Heart health Logistic regression was showing better results therefor these two were chosen for the final classification to be made. The results were found to be very promising and have scope for more research in the area.

## Block Diagram
![Picture1](https://user-images.githubusercontent.com/67185084/156569870-0d3f948b-6ccb-4e8e-bb2e-07a6830bea49.png)

### Cardiovascular Data
![Picture2 h vs nh](https://user-images.githubusercontent.com/67185084/156569944-6f9b4d08-f9a2-4e54-a99f-9aa28d2eccce.png)
![Picture3](https://user-images.githubusercontent.com/67185084/156569959-3e794542-e1a4-48b6-830c-32a7db0f3e43.png)
![corr](https://user-images.githubusercontent.com/67185084/156570043-23c90d44-bcb8-4d19-8da6-6967690a2d4c.png)
![pred](https://user-images.githubusercontent.com/67185084/156569978-a4395c79-7671-43aa-9aff-81a321a7c385.png)

### Breast Cancer Data
![Picture2](https://user-images.githubusercontent.com/67185084/156570301-e147304d-6f4a-4b96-8e1f-4f6db98aec44.png)
![Picture3](https://user-images.githubusercontent.com/67185084/156570313-a574514f-7ad4-448a-b58d-1e3a3a233753.png)
![Picture4](https://user-images.githubusercontent.com/67185084/156570330-a275bd59-07d4-44dd-9ede-03f752000397.png)
![Picture5](https://user-images.githubusercontent.com/67185084/156570350-fe4e0c62-35f4-47fa-b91e-3735828aa3bb.png)
![Picture6](https://user-images.githubusercontent.com/67185084/156570363-16d808f1-21c3-4faa-addc-dc42a791835c.png)

### UI of the website has two tabs one for Breast Cancer and one for Heart Health prdiction

![Picture7](https://user-images.githubusercontent.com/67185084/156570605-c4e257b2-ff8e-45fb-91a9-b0a6d7ec17a0.png)
![Picture8](https://user-images.githubusercontent.com/67185084/156570622-2984045e-bb11-4a6d-827b-4fb564ad2ae5.png)

