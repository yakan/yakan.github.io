---
layout: archive
permalink: /datascience/
title: "Data Science Projects by Categories"
author_profile: true
header:
  image: "/images/lake_tahoe.jpg"
---
{% for category in site.categories %}
  <h3>{{ category[0] }}</h3>
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}
