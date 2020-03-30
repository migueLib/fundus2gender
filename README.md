# Fundus2Sex

Fundus2Sex is a set of tools for Deep Learning analysis of fundus images. It also includes analysis and their results on the UKBiobank dataset. 

<p align="center">
	<img src="https://github.com/migueLib/fundus2gender/blob/master/figures/extras/fundus2gender.png">
</p>

Project Summary
--------

Deep convolutional neural networks (DCNN) excel in many medical classification tasks already. However, it is often unclear how DCNNs achieve this performance. Thus, extracting knowledge from artificial intelligence (AI) has become an important research topic.
 
An example for a puzzling achievement of an AI is the inference of gender/sex from fundus images, since ophthalmologists were not aware of significant anatomic differences between male and female retinae so far. We trained an inceptionv3 DCNN to classify gender/sex in fundus images from UKBiobank and reproduced results from Poplin et al., achieving an accuracy of 0.82 for gender/sex classification. We then tested different hypothesis and also screened for novel features by occlusion-sensitivity maps. By this means we were able to identify the angles between superior and inferior veins and arteries as an indicative feature for gender/sex classification. In the meantime Yamashita, et al. also came to a similar conclusion.
 
Artificial neural networks that were pretrained with images of a certain domain achieve higher discriminatory power for a classification taks of a related domain. Should we also expect this effect for biological neural networks? Let's find out and test! We launched a quizz at www.fundus2gender.org where you can participate. First, we will explain the rules how to infer the gender/sex from a fundus image. Then we will train you on 50 cases and finally we will test you.
 
Thanks for participating and we will report soon whether people with some prior experience in ophthalmology (experts) perform better than laymen!


Test Yourself Against AI
--------

Here: www.fundus2gender.org 


Get the trained InceptionV3 model
--------

Here: https://drive.google.com/open?id=1aCNeSgVINcjtlVBgiVD4CBoBrwWJxpWX


Check more reaserch from IGSB (Institute for Genomic Statistics and Bioinformatics)
--------

Here: https://www.igsb.uni-bonn.de/en
