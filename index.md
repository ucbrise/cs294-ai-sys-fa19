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
<blockquote class="blockquote">
<p>
The recent success of AI has been in large part due in part to advances in hardware and software systems. These systems have enabled training increasingly complex models on ever larger datasets. In the process, these systems have also simplified model development, enabling the rapid growth in the machine learning community. These new hardware and software systems include a new generation of GPUs and hardware accelerators (e.g., TPU and Nervana), open source frameworks such as Theano, TensorFlow, PyTorch, MXNet, Apache Spark, Clipper, Horovod, and Ray, and a myriad of systems deployed internally at companies just to name a few. 
At the same time, we are witnessing a flurry of ML/RL applications to improve hardware and system designs, job scheduling, program synthesis, and circuit layouts.  
</p>

<p>  
In this course, we will describe the latest trends in systems designs to better support the next generation of AI applications, and applications of AI to optimize the architecture and the performance of systems. 
The format of this course will be a mix of lectures, seminar-style discussions, and student presentations. 
Students will be responsible for paper readings, and completing a hands-on project. 
For projects, we will strongly encourage teams that contains both AI and systems students.
</p>
</blockquote>



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

This lecture will be an overview of the class, requirements, and an introduction to what makes great AI-Systems research.












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


</script>


