---
layout: default
---


# Machine Learning Systems (Fall 2019)

* **When**: *Mondays and Fridays from 2:00 to 3:30*
* **Where**: *Soda 310*
* **Instructor**: [Joseph E. Gonzalez](https://eecs.berkeley.edu/~jegonzal)
   * **Office Hours:** Wednesdays from 4:00 to 5:00 in 773 Soda Hall.
* **Announcements**: [Piazza](https://piazza.com/class/jz1seb32ovq4p2)
* **Sign-up to Present**: [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1EEf58o80eVCgPZmj71oBYRBcSVWsr7OMn9rOx6X9bnY/edit?usp=sharing) Every student should sign-up to present in at least three rows and as different roles each time.  Note that the Backup/Scribe presenter may be asked to fill in for one of the other roles with little notice.
* If you have reading suggestions please send a pull request to this course website on [Github](https://github.com/ucbrise/cs294-ai-sys-fa19) by modifying the [index.md](https://github.com/ucbrise/cs294-ai-sys-fa19/blob/master/index.md) file.



## Course Description

The recent success of AI has been in large part due in part to advances in hardware and software systems. These systems have enabled training increasingly complex models on ever larger datasets. In the process, these systems have also simplified model development, enabling the rapid growth in the machine learning community. These new hardware and software systems include a new generation of GPUs and hardware accelerators (e.g., TPU and Nervana), open source frameworks such as Theano, TensorFlow, PyTorch, MXNet, Apache Spark, Clipper, Horovod, and Ray, and a myriad of systems deployed internally at companies just to name a few. 
At the same time, we are witnessing a flurry of ML/RL applications to improve hardware and system designs, job scheduling, program synthesis, and circuit layouts.  

In this course, we will describe the latest trends in systems designs to better support the next generation of AI applications, and applications of AI to optimize the architecture and the performance of systems. 
The format of this course will be a mix of lectures, seminar-style discussions, and student presentations. 
Students will be responsible for paper readings, and completing a hands-on project. 
For projects, we will strongly encourage teams that contains both AI and systems students.



## New Course Format

A previous version of this course was offered in <a href="https://ucbrise.github.io/cs294-ai-sys-sp19/#today">Spring 2019</a>.  The format of this second offering is slightly different.  Each week will cover a different research area in AI-Systems.  The Monday lecture will be presented by Professor Gonzalez and will cover the context of the topic as well as a high-level overview of the reading for the week.  The Friday lecture will be organized around a mini program committee meeting for the weeks readings.  Students will be required to submit detailed reviews for a subset of the papers and lead the paper review discussions.  The goal of this new format is to both build a mastery of the material and also to develop a deeper understanding of how to evaluate and review research and hopefully provide insight into how to write better papers. 


## Course Syllabus



<!-- This is the dates for all the lectures -->
{% capture dates %}
8/30/19
9/2/19
9/6/19
9/9/19
9/13/19
9/16/19
9/20/19
9/23/19
9/27/19
9/30/19
10/4/19
10/7/19
10/11/19
10/14/19
10/18/19
10/21/19
10/25/19
10/28/19
11/1/19
11/4/19
11/8/19
11/11/19
11/15/19
11/18/19
11/22/19
11/25/19
11/29/19
12/2/19
12/6/19
12/9/19
12/13/19
12/16/19
12/20/19
{% endcapture %}
{% assign dates = dates | split: " " %}

This is a tentative schedule.  Specific readings are subject to change as new material is published.

<a href="#today"> Jump to Today </a>

<table class="table table-striped syllabus">
<thead>
   <tr>
      <th style="width: 5%"> Week </th>
      <th style="width: 10%"> Date (Lec.) </th>
      <th style="width: 85%"> Topic </th>
   </tr>
</thead>
<tbody>


{% include syllabus_entry %}
## Introduction and Course Overview

This lecture will be an overview of the class, requirements, and an introduction to the history of machine learning and systems research. 

* Lecture slides: [[pdf](assets/lectures/lec01/01_ai-sys-intro-small.pdf), [pptx](https://github.com/ucbrise/cs294-ai-sys-fa19/raw/master/assets/lectures/lec01/01_ai-sys-intro.pptx)]

<div class="reading">
<div class="optional_reading" markdown="1">

* [How to read a paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) provides some pretty good advice on how to read papers effectively.
* Timothy Roscoe's [writing reviews for systems conferences](https://people.inf.ethz.ch/troscoe/pubs/review-writing.pdf) will also help you in the reviewing process.

</div>
</div>






{% include syllabus_entry %}
# Holiday (Labor Day) 

There will be no class but please [sign-up](https://docs.google.com/spreadsheets/d/1EEf58o80eVCgPZmj71oBYRBcSVWsr7OMn9rOx6X9bnY/edit?usp=sharing) for the weekly discussion slots.







{% include syllabus_entry %}
## Big Ideas and How to Evaluate ML Systems Research


* [Submit your review](https://forms.gle/t6roBpS2QVPXn4zt6) before 1:00PM.
* Lecture slides: [[pdf](assets/lectures/lec02/02_ai-sys-big-ideas_v2.pdf), [pptx](https://github.com/ucbrise/cs294-ai-sys-fa19/raw/master/assets/lectures/lec02/02_ai-sys-big-ideas_v2.pptx)]


<!-- <div class="details" markdown="1"> 

Add more discussion

</div>
 -->

<div class="reading">
<div class="required_reading" markdown="1">

* [SysML: The New Frontier of Machine Learning Systems](https://arxiv.org/abs/1904.03257)
* Read Chapter 1 of [_Principles of Computer System Design_](https://www.sciencedirect.com/book/9780123749574/principles-of-computer-system-design). You will need to be on campus or use the Library VPN to obtain a free PDF.
* [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
* [A Berkeley View of Systems Challenges for AI](https://arxiv.org/pdf/1712.05855.pdf)


</div>
<div class="optional_reading" markdown="1">

### Additional Machine Learning Reading

* [Kevin Murphy's Textbook Introduction to Machine Learning](https://www.cs.ubc.ca/~murphyk/MLbook/pml-intro-22may12.pdf).  This provides a very high-level overview of machine learning.  You should probably know all of this. 
* [Stanford CS231n Tutorial on Neural Networks](http://cs231n.github.io/). I recommend reading Module 1 for a quick crash course in machine learning and some of the techniques used in this class.

### Additional Systems Reading

* [Hints for Computer System Design](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/acrobat-17.pdf)

### Open Debate about the Field

* Rich Sutton's [Post on Compute in ML](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) and the corresponding [Shimon Whiteson twitter debate](https://twitter.com/shimon8282/status/1106534178676506624?lang=en)


</div>
</div>







{% include syllabus_entry %}
## Machine Learning Life-cycle 

This lecture will discuss the machine learning life-cycle, spanning model development, training, and serving.  It will outline some of the technical machine learning and systems challenges at each stage and how these challenges interact.

* Lecture slides: [[pdf](assets/lectures/lec03/03_ml-lifecycle.pdf), [pptx](https://github.com/ucbrise/cs294-ai-sys-fa19/raw/master/assets/lectures/lec03/03_ml-lifecycle.pptx)]
* Template Slide Format for PC Meeting [[Google Drive](https://docs.google.com/presentation/d/1bxGL9cQziVhGQzhq-rBXxuYnuP3745Ye3PRg_1QZj3I/edit?usp=sharing)]


{% include syllabus_entry %}
## Discussion of Papers on Machine Learning Life-cycle


* [Submit your review](https://forms.gle/uRfHTWH86dee2WTd7) before 1:00PM.
* [Slides](https://docs.google.com/presentation/d/1s9x1fFcrUG0v-kQYtgG-yb0uaml-RuXQtb4g5lye-6o/edit#slide=id.p1) and [scribe notes](https://docs.google.com/document/d/1EoPWoTk3xb2Ii8tHdF9VIO3agMdvcGxhxcYcPfDbfFw/edit#heading=h.jfzr5y8bvswe) from the PC Meeting.
(These are only accessible to students enrolled in the class.)



<div class="reading">
<div class="required_reading" markdown="1">

* [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)
* [TFX: A TensorFlow-Based Production-Scale Machine Learning Platform](https://ai.google/research/pubs/pub46484.pdf)
* [Towards Unified Data and Lifecycle Management for Deep Learning](https://arxiv.org/pdf/1611.06224.pdf)

</div>
<div class="optional_reading" markdown="1">
* [Data Engineering Bulletin: Machine Learning Life-cycle Management](http://sites.computer.org/debull/A18dec/issue1.htm)
* [Context: The Missing Piece in the Machine Learning
Lifecycle](https://rlnsanz.github.io/dat/Flor_CMI_18_CameraReady.pdf)
* [Software 2.0 Blog Post](https://medium.com/@karpathy/software-2-0-a64152b37c35)
* [Doing Machine Learning the Uber Way: Five Lessons From the First Three Years of Michelangelo](https://towardsdatascience.com/doing-machine-learning-the-uber-way-five-lessons-from-the-first-three-years-of-michelangelo-da584a857cc2)
* [Introducing FBLearner Flow: Facebook’s AI backbone](https://engineering.fb.com/core-data/introducing-fblearner-flow-facebook-s-ai-backbone/)
* [DeepBird: Twitters ML Deployment Framework](https://blog.twitter.com/engineering/en_us/topics/insights/2018/twittertensorflow.html)
* [Demonstration of Mlflow: A System to Accelerate the Machine Learning Lifecycle](https://www.sysml.cc/doc/2019/demo_33.pdf)

### Software:
* [KubeFlow: Kubernetes Pipeline Orchestration Framework](https://www.kubeflow.org/)
* [MLflow](https://mlflow.org)


</div>
</div>







{% include syllabus_entry %}
## Database Systems and Machine Learning

In the previous lecture we saw that data and feature engineering is often the dominant hurtle in model development.  Database systems are often the source of data and the platform in which feature engineering takes place.  This lecture will cover some of the big ideas is database systems and how they relate to work on machine learning in databases.  


* Lecture slides: [[pdf](assets/lectures/lec04/04_learning_in_dbms.pdf), [pptx](https://github.com/ucbrise/cs294-ai-sys-fa19/raw/master/assets/lectures/lec04/04_learning_in_dbms.pptx)]
* Project Proposal [Sign-up doc](https://docs.google.com/document/d/1cEQ-rzsrSsdqa9dzcSuGCppxgN96eIuYT7AjpXrDqpo/edit#).  You must be enrolled in the class or on the waitlist to access this document.  Please add any projects you are thinking about starting and list yourself as interested in anyone else's projects.

{% include syllabus_entry %}
## Discussion of Database Systems and Machine Learning


* [Submit your review](https://forms.gle/BczjSGufNCoDekhp9) before 1:00PM.
* [Slides for PC Meeting](https://docs.google.com/presentation/d/1XCx2xBaaHgpAWAXpyTVwX16caMF3TZl6eWrKKTnNPRA/edit#slide=id.p1) posted. (These slides will only be accessible to students enrolled in the class.)

<div class="reading">
<div class="required_reading" markdown="1">

* [Towards a Unified Architecture for in-RDBMS Analytics](https://www.cs.stanford.edu/people/chrismre/papers/bismarck.pdf)
* [Materialization Optimizations for Feature Selection Workloads](https://cs.stanford.edu/people/chrismre/papers/mod539-zhang.pdf)
* [Learning Generalized Linear Models Over Normalized Data](http://pages.cs.wisc.edu/~jignesh/publ/GLMs-over-joins.pdf)


</div>
<div class="optional_reading" markdown="1">

* [Learning Linear Regression Models over Factorized Joins](http://www.cs.ox.ac.uk/dan.olteanu/papers/soc-sigmod16.pdf)
* [MauveDB: Supporting Model-based User Views in Database Systems](http://db.csail.mit.edu/pubs/sigmod06-mauvedb.pdf)
* [The MADlib Analytics Library or MAD Skills, the SQL](https://arxiv.org/pdf/1208.4165.pdf)

</div>
</div>











{% include syllabus_entry %}
## Machine Learning Frameworks and Automatic Differentiation

This week we will discuss recent development in model development and training frameworks.  While there is a long history of machine learning frameworks we will focus on frameworks for deep learning and automatic differentiation. In class we will review some of the big trends in machine learning framework design and basic ideas in forward and backward automatic differentiation.


* Lecture slides: [[pdf](assets/lectures/lec05/05_deep_learning_frameworks.pdf), [pptx](https://github.com/ucbrise/cs294-ai-sys-fa19/raw/master/assets/lectures/lec05/05_deep_learning_frameworks.pptx)]


_Project proposals are due next Monday_


{% include syllabus_entry %}
## Machine Learning Frameworks and Automatic Differentiation

_Update:_ Two of the readings were changed to reflect a focus on deep learning frameworks. The previous readings on SystemML and KeystoneML have been moved to optional reading. 

* [Submit your review](https://forms.gle/PThRMUEReqEut86q6) before 1:00PM.
* [Slides for PC Meeting](https://docs.google.com/presentation/d/10bqrri3CbX6eaqdzRQDGU-yJAVyLLVYVh3k93DwXexw/edit#slide=id.p1) ] These slides will only be accessible to students enrolled in the class.




<div class="reading">
<div class="required_reading" markdown="1">

* [Automatic differentiation in ML: Where we are and where we should be going](https://papers.nips.cc/paper/8092-automatic-differentiation-in-ml-where-we-are-and-where-we-should-be-going)
* [TensorFlow: A System for Large-Scale Machine Learning](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)
* [JANUS: Fast and Flexible Deep Learning via Symbolic Graph Execution of Imperative Programs](https://arxiv.org/pdf/1812.01329.pdf)

</div>
<div class="optional_reading" markdown="1">

### Pipeline Training Frameworks (Classical)
* [KeystoneML: Optimizing Pipelines for Large-Scale
Advanced Analytics](https://shivaram.org/publications/keystoneml-icde17.pdf)
* [SystemML: Declarative Machine Learning on Spark](http://www.vldb.org/pvldb/vol9/p1425-boehm.pdf)

### Automatic Differentiation and Differentiable Programming
* [Automatic Differentiation in Machine Learning: a Survey](http://www.jmlr.org/papers/volume18/17-468/17-468.pdf)
* [Roger Grosse's Lecture Notes on Automatic Differentiation](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L06%20Automatic%20Differentiation.pdf)
* [A Differentiable Programming System to Bridge Machine Learning and Scientific Computing](https://arxiv.org/pdf/1907.07587.pdf)


### Deep Learning Frameworks with Automatic Differentiation
* [Caffe: Convolutional Architecture for Fast Feature Embedding](https://arxiv.org/pdf/1408.5093.pdf)
* [Theano: A Python Framework for Fast Computation of Mathematical Expressions](https://arxiv.org/pdf/1605.02688.pdf) and [Theano: A CPU and GPU Math Compiler in Python](http://www.iro.umontreal.ca/~lisa/pointeurs/theano_scipy2010.pdf)
* [Automatic differentiation in PyTorch](https://openreview.net/pdf?id=BJJsrmfCZ)
* [MXNet: A Flexible and Efficient Machine Learning
Library for Heterogeneous Distributed Systems](https://arxiv.org/pdf/1512.01274.pdf)
* [TensorFlow Eager: A Multi-Stage, Python-Embedded DSL for Machine Learning](https://arxiv.org/pdf/1903.01855.pdf)
* [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://arxiv.org/pdf/1603.04467.pdf)


### Deep Learning Primitives
* [cuDNN: Efficient Primitives for Deep Learning](https://arxiv.org/pdf/1410.0759.pdf)


</div>
</div>







{% include syllabus_entry %}
## Distributed Model Training


This week we will discuss developments in distributed training.  We will quickly review the statistical query model pushed by early map-reduce machine learning frameworks and then discuss advances in parameter servers and distributed neural network training.

* Lecture slides: [[pdf](assets/lectures/lec06/06_distributed_training.pdf), [pptx](https://github.com/ucbrise/cs294-ai-sys-fa19/raw/master/assets/lectures/lec06/06_distributed_training.pptx)]


### Project Proposals Due!
* One Page Project description due at 11:59 PM.  Check out the [suggested projects](https://docs.google.com/document/d/1cEQ-rzsrSsdqa9dzcSuGCppxgN96eIuYT7AjpXrDqpo/edit#).  Submit a link to your one page Google document containing your project descriptions [to this google form](https://forms.gle/buSyKy7anciNgLqNA).  You only need one submission per team but please list all the team member's email addresses.  You can also update your submission if needed.



{% include syllabus_entry %}
## Discussion on Distributed Model Training


* [Submit your review](https://forms.gle/foiDoWZs1cRqqnhF7) before 1:00PM.
* [Slides for PC Meeting](https://docs.google.com/presentation/d/1lGq3nljQqKffqAoLJt8GXQ-fsmTIT_UVmtGzQ2LZJM0/edit#slide=id.p1) (These slides will only be accessible to students enrolled in the class.)



<div class="reading">
<div class="required_reading" markdown="1">

* [Scaling Distributed Machine Learning with the Parameter Server](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf)
* [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://cs.stanford.edu/~matei/papers/2019/sosp_pipedream.pdf)
* [Adaptive Communication Strategies to Achieve the Best Error-Runtime Trade-off in Local-Update SGD](https://arxiv.org/pdf/1810.08313.pdf) 


</div>
<div class="optional_reading" markdown="1">

* [Demystifying Parallel and Distributed Deep Learning: An
In-Depth Concurrency Analysis](https://arxiv.org/pdf/1802.09941.pdf)
* [Integrated Model, Batch, and Domain Parallelism in Training Neural Networks](https://arxiv.org/pdf/1712.04432.pdf)
* [Effect of batch size on training dynamics](https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e)
* [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://openreview.net/pdf?id=SkhQHMW0W)
* [Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)[[pdf](assets/lectures/lec17/hogwild_final.pdf)]
* [Large Scale Distributed Deep Networks](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf)
* [Scaling Distributed Machine Learning with In-Network Aggregation](https://arxiv.org/pdf/1903.06701.pdf)

### ImageNet in X Minutes
* [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
* [Now anyone can train Imagenet in 18 minutes](https://www.fast.ai/2018/08/10/fastai-diu-imagenet/)
* [Highly Scalable Deep Learning Training System with
Mixed-Precision: Training ImageNet in Four Minutes](https://arxiv.org/pdf/1807.11205.pdf)


### All-Reduce
* [Baidu Ring All-Reduce Blog Post ](http://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)
* The original Ring All-Reduce Paper ["Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations"](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf)
* [Visual intuition on ring-Allreduce for distributed Deep Learning](https://towardsdatascience.com/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da)
* [Double Binary Trees](https://devblogs.nvidia.com/massively-scale-deep-learning-training-nccl-2-4/#ref6)



</div>
</div>














{% include syllabus_entry %}
## Prediction Serving

Until recently, much of the focus on systems research was aimed at model training.  However, recently there has been a growing interest in addressing the challenges of prediction serving.  This lecture will frame the challenges of prediction serving and cover some of the recent advances. 


* Lecture slides: [[pdf](assets/lectures/lec07/07_prediction-serving.pdf), [pptx](https://github.com/ucbrise/cs294-ai-sys-fa19/raw/master/assets/lectures/lec07/07_prediction-serving_highres.pptx)]



{% include syllabus_entry %}
## Power Outage Related Holiday

Unfortunately, class was canceled and so the PC Meeting has been moved to Monday.   Note that early project presentations are also due next Friday.






{% include syllabus_entry %}
## Discussion on Prediction Serving

* [Submit your review](https://forms.gle/Fs5HXiM6msjPAEMz8) before 1:00PM.
* [Slides for PC Meeting](https://docs.google.com/presentation/d/1bT_9e85gmC8i271IpbjFz97eemWYDp3UdyXHG2XGMlM/edit#slide=id.p1) (These slides will only be accessible to students enrolled in the class.)


<div class="reading">
<div class="required_reading" markdown="1">

* [Pretzel: Opening the Black Box of Machine Learning Prediction Serving Systems](https://www.usenix.org/system/files/osdi18-lee.pdf)
* [InferLine: ML Inference Pipeline Composition Framework](assets/preprint/inferline_draft.pdf) (pre-print)
* [Focus: Querying Large Video Datasets with Low Latency and Low Cost](https://www.usenix.org/system/files/osdi18-hsieh.pdf)

</div>
<div class="optional_reading" markdown="1">

The [Prediction-Serving Systems: What happens when we wish to actually deploy a machine learning model to production?](https://queue.acm.org/detail.cfm?id=3210557) ACM Queue article provides a nice overview.

#### Systems Reading:
* [Live Video Analytics at Scale with Approximation and Delay-Tolerance](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-zhang.pdf)
* [LASER: A Scalable Response Prediction Platform For Online Advertising](https://dorx.me/papers/p173-agarwal.pdf)
* [TensorFlow-Serving: Flexible, High-Performance ML Serving](https://arxiv.org/abs/1712.06139)
* [Clipper: A Low-Latency Online Prediction Serving System](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf)
* [Deep Learning Inference in Facebook Data Centers: Characterization, Performance Optimizations and Hardware Implications](https://arxiv.org/abs/1811.09886)
* [The Missing Piece in Complex Analytics: Low Latency, Scalable Model Management and Serving with Velox](http://arxiv.org/abs/1409.3809)
* [The Case for Predictive Database Systems: Opportunities and Challenges](http://cidrdb.org/cidr2011/Papers/CIDR11_Paper20.pdf). 

#### More Efficient Models:

* Paul Viola and Michael Jones [Rapid Object Detection using a Boosted Cascade of Simple Features](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) CVPR 2001. 

#### Performance Breakdown of various models
* [Benchmark Analysis of Representative Deep Neural Network Architectures](https://arxiv.org/pdf/1810.00736.pdf)

</div>
</div>










{% include syllabus_entry %}
# Project Presentations


{% include syllabus_entry %}
## Finish Project Presentations and Start Model Compilation

This week we will explore the process of compiling/optimizing deep neural network computation graphs.  This reading will span both graph level optimization as well as the compilation and optimization of individual tensor operations.  


* Lecture slides: [[pdf](assets/lectures/lec08/08_dl_compilers.pdf), [pptx](https://github.com/ucbrise/cs294-ai-sys-fa19/raw/master/assets/lectures/lec08/08_dl_compilers.pptx)]






{% include syllabus_entry %}
## Discussion of Model Compilation

* [Submit your review](https://forms.gle/pv6FaRfh1NmYYwv76) before 1:00PM.
* [Slides for PC Meeting](https://docs.google.com/presentation/d/1pkH4LMeCanUqIp3KS_3sJZw5P3_MBtcXB2Ff3mSjiDA/edit#slide=id.p1) (These slides will only be accessible to students enrolled in the class.)




<div class="reading">
<div class="required_reading" markdown="1">

* [Optimizing DNN Computation with Relaxed Graph Substitutions](https://www.sysml.cc/doc/2019/22.pdf)
* [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)
* [Learning to Optimize Halide with Tree Search and Random Programs](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf)

</div>
<div class="optional_reading" markdown="1">

* [Learning to Optimize Tensor Programs](https://arxiv.org/abs/1805.08166): The TVM story is two fold. There's a System for ML story (above paper) and this paper is their the ML for System story.
* [Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks](http://proceedings.mlr.press/v80/jia18a/jia18a.pdf)
* [TensorComprehensions](https://arxiv.org/abs/1802.04730)
* [Supporting Very Large Models using Automatic Dataflow Graph Partitioning](https://arxiv.org/pdf/1807.08887.pdf)


</div>
</div>





{% include syllabus_entry %}
## PG&amp;E and Fire Related Cancellation

Unfortunately, due to the power outage, lecture is canceled today.  To make up for lost lecture(s) and accommodate our guest speakers, we will skip the overview lecture this week and start with the PC meeting on Machine Learning Applied to Systems.  However, this will put a little extra pressure on the neutral presenters to provide additional context.  We will then cover the discussion on machine learning hardware the following Monday.




{% include syllabus_entry %}
## Discussion of Machine Learning Applied to Systems Day 1


* [Submit your review](https://forms.gle/6GEdrphbLAamDhCFA) before 1:00PM.
* [Slides for PC Meeting](https://docs.google.com/presentation/d/1_Ch1N_IaI-cRXwKh5t7AdjeIahcqbriL1aIrLWGxF2E/edit#slide=id.p1) (These slides will only be accessible to students enrolled in the class.)




<div class="reading">
<div class="required_reading" markdown="1">

* [Resource Central: Understanding and Predicting Workloads for Improved Resource Management in Large Cloud Platforms](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/10/Resource-Central-SOSP17.pdf)
* [Device Placement Optimization with Reinforcement Learning](https://arxiv.org/pdf/1706.04972.pdf)
* [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208)

</div>
<div class="optional_reading" markdown="1">

* [Quasar: Resource-Efficient and QoS-Aware Cluster Management](http://www.csl.cornell.edu/~delimitrou/papers/2014.asplos.quasar.pdf)


</div>
</div>





{% include syllabus_entry %}
## Hardware Acceleration for Machine Learning

This lecture will be presented by [Kurt Keutzer](https://people.eecs.berkeley.edu/~keutzer/) and [Suresh Krishna](https://www.linkedin.com/in/suresh-krishna-793506158) who are experts in processor design as well as network and architecture co-design.


* Guest lecture slides: [[pdf](assets/lectures/lec09/2019-11-04-1000am-sysml-class-kk.pdf), [pptx](https://github.com/ucbrise/cs294-ai-sys-fa19/raw/master/assets/lectures/lec09/2019-11-04-1000am-sysml-class-kk.pptx)]



{% include syllabus_entry %}
## Discussion Hardware Acceleration for Machine Learning


* [Submit your review](https://forms.gle/1TkxN2KKyqL2mWFHA) before 1:00PM.
* [Slides for PC Meeting](https://docs.google.com/presentation/d/1DsL9kSnKHlFp04vDmT6wEbxlvXsCIpbw8zMnfZaa5mQ/edit#slide=id.p1) (These slides will only be accessible to students enrolled in the class.)





<div class="reading">
<div class="required_reading" markdown="1">

* [A Configurable Cloud-Scale DNN Processor for Real-Time AI](https://www.microsoft.com/en-us/research/uploads/prod/2018/06/ISCA18-Brainwave-CameraReady.pdf)
* [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/pdf/1704.04760.pdf)
* [Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks](http://www.rle.mit.edu/eems/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf)


</div>
<div class="optional_reading" markdown="1">

* [Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://arxiv.org/pdf/1703.09039.pdf)
* A great [spreadsheet analysis](https://docs.google.com/spreadsheets/u/1/d/1xAo6TcSgHdd25EdQ-6GqM0VKbTYu8cWyycgJhHRVIgY/edit?usp=sharing) of the power and performance characteristics of all the publicly available hardware accelerators for deep learning (GPUs, CPU, TPUs).
* Nvidia [post](https://developer.nvidia.com/deep-learning-performance-training-inference) comparing different GPUs across a wide range of networks.


</div>
</div>


 








{% include syllabus_entry %}
# (11/11) Administrative Holiday


{% include syllabus_entry %}
## Discussion of Machine Learning Applied to Systems Day 2



* [Submit your review](https://forms.gle/i4UJfqFNV9376yGZ9) before 1:00PM.
* [Slides for PC Meeting](https://docs.google.com/presentation/d/1ZAgtOBk3DWCEnFvvsvaXQ2Mu8C32MfGS_zoQhUqQ6xo/edit#slide=id.p1) coming soon. (These slides will only be accessible to students enrolled in the class.)




<div class="reading">
<div class="required_reading" markdown="1">

* [AuTO: Scaling Deep Reinforcement Learning to Enable Datacenter-Scale Automatic Traffic Optimization](https://conferences.sigcomm.org/events/apnet2018/papers/auto.pdf)
* [Neural Adaptive Video Streaming with Pensieve](https://people.csail.mit.edu/hongzi/content/publications/Pensieve-Sigcomm17.pdf)  
* [Neural Adaptive Content-aware Internet Video Delivery](https://www.usenix.org/system/files/osdi18-yeo.pdf)

</div>
<div class="optional_reading" markdown="1">


</div>
</div>



{% include syllabus_entry %}
## Learning with Adversaries

This week we will discuss machine learning in adversarial settings.  This includes secure federated learning, differential privacy, and adversarial examples.


* Lecture slides: [[pdf](assets/lectures/lec10/10_adversarial_ml.pdf), [pptx](https://github.com/ucbrise/cs294-ai-sys-fa19/raw/master/assets/lectures/lec10/10_adversarial_ml.pptx)]





{% include syllabus_entry %}
## Discussion on Learning with Adversaries


* [Submit your review](https://forms.gle/fAzS9529uss58Q1q9) before 1:00PM.
* [Slides for PC Meeting](https://docs.google.com/presentation/d/1J96RDrfxHUC8wyDcthKGrEeYTys9-jKXVlvLnlNmT1Q/edit#slide=id.p1) coming soon. (These slides will only be accessible to students enrolled in the class.)



<div class="reading">
<div class="required_reading" markdown="1">

* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)
* [Privacy Accounting and Quality Control in the Sage Differentially Private ML Platform](https://arxiv.org/pdf/1909.01502.pdf)  
* [Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware](https://arxiv.org/abs/1806.03287)

</div>
<div class="optional_reading" markdown="1">

* [Helen: Maliciously Secure Coopetitive Learning for Linear Models](https://people.eecs.berkeley.edu/~wzheng/helen_ieeesp.pdf)
* [Faster CryptoNets: Leveraging Sparsity for Real-World Encrypted Inference](https://arxiv.org/abs/1811.09953)
* [Rendered Insecure: GPU Side Channel Attacks are Practical](https://www.cs.ucr.edu/~zhiyunq/pub/ccs18_gpu_side_channel.pdf)
* [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
* [Federated Learning: Collaborative Machine Learning without Centralized Training Data](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
* [Federated Learning at Google ... A comic strip?](https://federated.withgoogle.com)
* [SecureML: A System for Scalable Privacy-Preserving Machine Learning](https://eprint.iacr.org/2017/396.pdf)
* More reading coming soon ...


</div>
</div>






{% include syllabus_entry %}
## Autonomous Driving

Autonomous vehicles will likely transform society in the next decade and are fundamentally AI enabled systems.  In this lecture we will discuss the AI-Systems challenges around autonomous driving.


* Lecture slides: [[pdf](assets/lectures/lec11/11_adversarial_ml.pdf), [pptx](https://github.com/ucbrise/cs294-ai-sys-fa19/raw/master/assets/lectures/lec11/11_autonomous_driving.pptx)]



    



{% include syllabus_entry %}
# (11/29) Holiday (Thanksgiving)




{% include syllabus_entry %}
## Discussion on Autonomous Driving

Everyone must do one of the readings (you pick).     



* [Submit your review](https://forms.gle/i47umoVaVSWjV7uL9) before 1:00PM.
* [Slides for PC Meeting](https://docs.google.com/presentation/d/1kq8nOqVjQAVHIFcLiwqo8oFay0hRXcdLJuZIkabIsb0/edit#slide=id.p1) coming soon. (These slides will only be accessible to students enrolled in the class.)


<div class="reading">
<div class="required_reading" markdown="1">

* [Self-Driving Cars: A Survey](https://arxiv.org/abs/1901.04407).  This is a slightly longer survey so focus more on the overview and framing first few pages of the autonomous driving problem and common solutions.  
* [The Architectural Implications of Autonomous Driving: Constraints and Acceleration](https://web.eecs.umich.edu/~shihclin/papers/AutonomousCar-ASPLOS18.pdf)
* [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/abs/1812.03079)


</div>
<div class="optional_reading" markdown="1">

* [An Open Approach to Autonomous Vehicles](https://ieeexplore.ieee.org/document/7368032 )
* [End-to-End Learning of Driving Models with Surround-View Cameras and Route Planners](https://arxiv.org/abs/1803.10158)

### DARPA Grand Challenges
* [Software Infrastructure for an Autonomous Ground Vehicle](https://www.ri.cmu.edu/pub_files/2008/12/TartanInfrastructure.pdf)
* [Stanley: The Robot that Won the DARPA Grand Challenge](https://dl.acm.org/citation.cfm?id=1210482)
* [Tartan Racing: A Multi-Modal Approach to the DARPA Urban Challenge](https://www.ri.cmu.edu/pub_files/2007/4/Tartan_Racing.pdf)
* [Towards a Viable Autonomous Driving Research Platform](https://www.ri.cmu.edu/pub_files/2013/6/2013_IV_SRX.pdf)
* [Engineering Autonomous Driving Software](https://arxiv.org/pdf/1409.6579.pdf)

</div>
</div>


{% include syllabus_entry %}
## Conclusion!


{% include syllabus_entry %}
# (12/6) RRR Week

{% include syllabus_entry %}
# (12/9) RRR Week

{% include syllabus_entry %}
# (12/16) Poster Presentations 

{% include syllabus_entry %}
# (12/20) No Class

Don't forget to submit your final reports.  As noted on Piazza, the final report should be 6-pages plus references (2-column, 10pt font, unlimited appendix).  Please submit your report using this form:

<center>
<h3>
    <a href="https://forms.gle/tLcLeEzRzueFGUcG6">Submit Your Report Here</a>
</h3>
</center>

You only need to submit the project once per team.  The write-up should discuss the problem formulation, related work, your approach, and your results.  





</td>
</tr>
</tbody>
</table>






## Projects

Detailed candidate project descriptions will be posted shortly.  However, students are encourage to find projects that relate to their ongoing research.


## Grading

Grades will be largely based on class participation and projects.  In addition, we will require weekly paper summaries submitted before class.
* **Projects:** _60%_
* **Weekly Summaries:** _20%_
* **Class Participation:** _20%_









<script type="text/javascript">


var current_date = new Date();
var rows = document.getElementsByTagName("th");
var finished =  false;
for (var i = 1; i < rows.length && !finished; i++) {
   var r = rows[i];
   if (r.id.startsWith("counter_")) {
      var fields = r.id.split("_")
      var week_div_id = "week_" + fields[2]
      var lecture_date = new Date(fields[1] + " 23:59:00")
      if (current_date <= lecture_date) {
         finished = true;
         r.style.background = "orange"
         r.style.color = "black"
         var week_td = document.getElementById(week_div_id)
         week_td.style.background = "#043361"
         week_td.style.color = "white"
         var anchor = document.createElement("div")
         anchor.setAttribute("id", "today")
         week_td.prepend(anchor)
      }
   }
}

$(".reading").each(function(ind, elem) {
   var optional_reading = $(elem).find(".optional_reading");
   if(optional_reading.length == 1) {
      optional_reading = optional_reading[0];
      optional_reading.setAttribute("id", "optional_reading_" + ind);
      var button = document.createElement("button");
      button.setAttribute("class", "btn btn-primary btn-sm");
      button.setAttribute("type", "button");
      button.setAttribute("data-toggle", "collapse");
      button.setAttribute("data-target", "#optional_reading_" + ind);
      button.setAttribute("aria-expanded", "false");
      button.setAttribute("aria-controls", "#optional_reading_" + ind);
      optional_reading.setAttribute("class", "optional_reading_no_heading collapse")
      button.innerHTML = "Additional Optional Reading";
      optional_reading.before(button)
   }
   
})


$(".details").each(function(ind, elem) {
      elem.setAttribute("id", "details_" + ind);
      var button = document.createElement("button");
      button.setAttribute("class", "btn btn-primary btn-sm");
      button.setAttribute("type", "button");
      button.setAttribute("data-toggle", "collapse");
      button.setAttribute("data-target", "#details_" + ind);
      button.setAttribute("aria-expanded", "false");
      button.setAttribute("aria-controls", "#details_" + ind);
      elem.setAttribute("class", "details_no_heading collapse")
      button.innerHTML = "Detailed Description";
      elem.before(button)
   })

</script>


