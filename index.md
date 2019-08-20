---
layout: default
---


# Machine Learning Systems (Fall 2019)

* **When**: *Mondays and Fridays from 2:00 to 3:30*
* **Where**: *Soda 310*
* **Instructor**: [Joseph E. Gonzalez](https://eecs.berkeley.edu/~jegonzal)
* **Announcements**: [Piazza](https://piazza.com/class/jz1seb32ovq4p2)

<!-- * **Sign-up to Present**: [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1NLLVPh8QioXRtzYEKc3XjtJMLqbT8WMMQ27bQz8lSJI/edit?usp=sharing
)
* **Project Ideas**: [Google Spreadsheet](https://docs.google.com/spreadsheets/d/16Oz8ZJ0x1AdukWQxq7QYdzkzoVH70vbhtSOGlJ_EFKc/edit#gid=0)
* If you have reading suggestions please send a pull request to this course website on [Github](https://github.com/ucbrise/cs294-ai-sys-sp19) by modifying the [index.md](https://github.com/ucbrise/cs294-ai-sys-sp19/blob/master/index.md) file.
 -->


## Course Description

The recent success of AI has been in large part due in part to advances in hardware and software systems. These systems have enabled training increasingly complex models on ever larger datasets. In the process, these systems have also simplified model development, enabling the rapid growth in the machine learning community. These new hardware and software systems include a new generation of GPUs and hardware accelerators (e.g., TPU and Nervana), open source frameworks such as Theano, TensorFlow, PyTorch, MXNet, Apache Spark, Clipper, Horovod, and Ray, and a myriad of systems deployed internally at companies just to name a few. 
At the same time, we are witnessing a flurry of ML/RL applications to improve hardware and system designs, job scheduling, program synthesis, and circuit layouts.  

In this course, we will describe the latest trends in systems designs to better support the next generation of AI applications, and applications of AI to optimize the architecture and the performance of systems. 
The format of this course will be a mix of lectures, seminar-style discussions, and student presentations. 
Students will be responsible for paper readings, and completing a hands-on project. 
For projects, we will strongly encourage teams that contains both AI and systems students.



## Updated Course Format

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


{% include syllabus_entry %}
# Holiday (Labor Day) 

There will be no class but please sign up for the weekly discussion slots.







{% include syllabus_entry %}
## Big Ideas and How to Evaluate ML Systems Research

This lecture will review the big concepts in systems and machine learning and then discuss how to evaluate ML Systems research.  

<!-- <div class="details" markdown="1"> 

something somehting 

</div>
 -->

<div class="reading">
<div class="required_reading" markdown="1">

* [SysML: The New Frontier of Machine Learning Systems](https://arxiv.org/abs/1904.03257)
* Read Chapter 1 of [_Principles of Computer System Design_](https://www.sciencedirect.com/book/9780123749574/principles-of-computer-system-design). You will need to be on campus or use the Library VPN to obtain a free PDF.
* [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
* [A Berkeley View of Systems Challenges for AI](A Berkeley View of Systems Challenges for AI)


</div>
<div class="optional_reading" markdown="1">

### Additional Machine Learning Reading

* [Kevin Murphy's Textbook Introduction to Machine Learning](https://www.cs.ubc.ca/~murphyk/MLbook/pml-intro-22may12.pdf).  This provides a very high-level overview of machine learning.  You should probably know all of this. 
* [Principles of Computer System Design: An Introduction](https://ocw.mit.edu/resources/res-6-004-principles-of-computer-system-design-an-introduction-spring-2009/online-textbook/part_ii_open_5_0.pdf).  Chapter 1 of this book gives a good summary of Lampson's article.
* [Stanford CS231n Tutorial on Neural Networks](http://cs231n.github.io/). I recommend reading Module 1 for a quick crash course in machine learning and some of the techniques used in this class.

### Additional Systems Reading

* [Hints for Computer System Design](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/acrobat-17.pdf)


</div>
</div>





{% include syllabus_entry %}
## Machine Learning Life-cycle 

{% include syllabus_entry %}
## Discussion of Papers on Machine Learning Life-cycle



{% include syllabus_entry %}
## Database Systems and Machine Learning

{% include syllabus_entry %}
## Discussion of Database Systems and Machine Learning



{% include syllabus_entry %}
## Prediction Serving

{% include syllabus_entry %}
## Discussion of Prediction Serving




{% include syllabus_entry %}
## Model Development and Training Frameworks

{% include syllabus_entry %}
## Discussion of Model Development and Training Frameworks



{% include syllabus_entry %}
## Distributed Model Training

{% include syllabus_entry %}
## Discussion of Distributed Model Training



{% include syllabus_entry %}
## Application Area: Autonomous Driving

{% include syllabus_entry %}
# Project Presentations


{% include syllabus_entry %}
## Model Compilation

{% include syllabus_entry %}
## Discussion of Model Compilation




{% include syllabus_entry %}
## Hardware Acceleration for Machine Learning

{% include syllabus_entry %}
## Discussion Hardware Acceleration for Machine Learning





{% include syllabus_entry %}
## Machine Learning Applied to Systems

{% include syllabus_entry %}
## Discussion of Machine Learning Applied to Systems







{% include syllabus_entry %}
# (11/11) Administrative Holiday

{% include syllabus_entry %}
## TBD



{% include syllabus_entry %}
## Secure / Coopetitive Machine Learning

{% include syllabus_entry %}
## Discussion on Secure ML





{% include syllabus_entry %}
## TBD


{% include syllabus_entry %}
# (11/29) Holiday (Thanksgiving)





{% include syllabus_entry %}
## Conclusion!


{% include syllabus_entry %}
# (12/6) RRR Week

{% include syllabus_entry %}
# (12/9) RRR Week

{% include syllabus_entry %}
# (12/6) Exams

{% include syllabus_entry %}
# (12/16) Poster Presentation




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


