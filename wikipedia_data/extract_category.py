import re
import json

# this files extracts the categories from the html of the wikipedia page

raw_html = ["""
<b><a href="/wiki/Category:Formal_sciences" title="Category:Formal sciences">Formal sciences</a></b>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="margin-left:0em; padding: 0.5em 0">
<ul><li><b><a href="/wiki/Category:Mathematics" title="Category:Mathematics">Mathematics</a></b></li>
<li><a href="/wiki/Category:Mathematics_education" title="Category:Mathematics education">Mathematics education</a></li>
<li><a href="/wiki/Category:Equations" title="Category:Equations">Equations</a></li>
<li><a href="/wiki/Category:Heuristics" title="Category:Heuristics">Heuristics</a></li>
<li><a href="/wiki/Category:Measurement" title="Category:Measurement">Measurement</a></li>
<li><a href="/wiki/Category:Numbers" title="Category:Numbers">Numbers</a></li>
<li><a href="/wiki/Category:Mathematical_proofs" title="Category:Mathematical proofs">Proofs</a></li>
<li><a href="/wiki/Category:Theorems" title="Category:Theorems">Theorems</a></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="margin-left:1.0em; padding: 0.5em 0;">
<ul><li><b><a href="/wiki/Category:Fields_of_mathematics" title="Category:Fields of mathematics">Fields of mathematics</a></b></li>
<li><a href="/wiki/Category:Arithmetic" title="Category:Arithmetic">Arithmetic</a></li>
<li><a href="/wiki/Category:Algebra" title="Category:Algebra">Algebra</a></li>
<li><a href="/wiki/Category:Geometry" title="Category:Geometry">Geometry</a></li>
<li><a href="/wiki/Category:Trigonometry" title="Category:Trigonometry">Trigonometry</a></li>
<li><a href="/wiki/Category:Mathematical_analysis" title="Category:Mathematical analysis">Mathematical analysis</a></li>
<li><a href="/wiki/Category:Calculus" title="Category:Calculus">Calculus</a></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="margin-left:0em; padding: 0.5em 0;">
<ul><li><b><a href="/wiki/Category:Logic" title="Category:Logic">Logic</a></b></li>
<li><a href="/wiki/Category:Deductive_reasoning" title="Category:Deductive reasoning">Deductive reasoning</a></li>
<li><a href="/wiki/Category:Inductive_reasoning" title="Category:Inductive reasoning">Inductive reasoning</a></li>
<li><a href="/wiki/Category:History_of_logic" title="Category:History of logic">History of logic</a></li>
<li><a href="/wiki/Category:Fallacies" title="Category:Fallacies">Fallacies</a></li>
<li><a href="/wiki/Category:Metalogic" title="Category:Metalogic">Metalogic</a></li>
<li><a href="/wiki/Category:Philosophy_of_logic" title="Category:Philosophy of logic">Philosophy of logic</a></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="margin-left:0em; padding: 0.5em 0;">
<ul><li><b><a href="/wiki/Category:Mathematical_sciences" title="Category:Mathematical sciences">Mathematical sciences</a></b></li>
<li><a href="/wiki/Category:Computational_science" title="Category:Computational science">Computational science</a></li>
<li><a href="/wiki/Category:Operations_research" title="Category:Operations research">Operations research</a></li>
<li><a href="/wiki/Category:Theoretical_physics" title="Category:Theoretical physics">Theoretical physics</a></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="margin-left:1.0em; padding: 0.5em 0;">
<ul><li><b><a href="/wiki/Category:Statistics" title="Category:Statistics">Statistics</a></b></li>
<li><a href="/wiki/Category:Analysis_of_variance" title="Category:Analysis of variance">Analysis of variance</a></li>
<li><a href="/wiki/Category:Bayesian_statistics" title="Category:Bayesian statistics">Bayesian statistics</a></li>
<li><a href="/wiki/Category:Categorical_data" title="Category:Categorical data">Categorical data</a></li>
<li><a href="/wiki/Category:Covariance_and_correlation" title="Category:Covariance and correlation">Covariance and correlation</a></li>
<li><a href="/wiki/Category:Data_analysis" title="Category:Data analysis">Data analysis</a></li>
<li><a href="/wiki/Category:Decision_theory" title="Category:Decision theory">Decision theory</a></li>
<li><a href="/wiki/Category:Design_of_experiments" title="Category:Design of experiments">Design of experiments</a></li>
<li><a href="/wiki/Category:Logic_and_statistics" title="Category:Logic and statistics">Logic and statistics</a></li>
<li><a href="/wiki/Category:Multivariate_statistics" title="Category:Multivariate statistics">Multivariate statistics</a></li>
<li><a href="/wiki/Category:Non-parametric_statistics" title="Category:Non-parametric statistics">Non-parametric statistics</a></li>
<li><a href="/wiki/Category:Parametric_statistics" title="Category:Parametric statistics">Parametric statistics</a></li>
<li><a href="/wiki/Category:Regression_analysis" title="Category:Regression analysis">Regression analysis</a></li>
<li><a href="/wiki/Category:Sampling_(statistics)" title="Category:Sampling (statistics)">Sampling</a></li>
<li><a href="/wiki/Category:Statistical_theory" title="Category:Statistical theory">Statistical theory</a></li>
<li><a href="/wiki/Category:Stochastic_processes" title="Category:Stochastic processes">Stochastic processes</a></li>
<li><a href="/wiki/Category:Summary_statistics" title="Category:Summary statistics">Summary statistics</a></li>
<li><a href="/wiki/Category:Survival_analysis" title="Category:Survival analysis">Survival analysis</a></li>
<li><a href="/wiki/Category:Time_series" title="Category:Time series">Time series</a></li></ul>
</div>
""",
            """<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1033289096"><div role="note" class="hatnote navigation-not-searchable">Main categories: <a href="/wiki/Category:Science" title="Category:Science">Science</a>, <a href="/wiki/Category:Natural_sciences" title="Category:Natural sciences">Natural sciences</a> and <a href="/wiki/Category:Nature" title="Category:Nature">Nature</a></div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="padding: 0.5em 0">
<ul><li><b><a href="/wiki/Category:Biology" title="Category:Biology">Biology</a></b></li>
<li><a href="/wiki/Category:Botany" title="Category:Botany">Botany</a></li>
<li><a href="/wiki/Category:Ecology" title="Category:Ecology">Ecology</a></li>
<li><a href="/wiki/Category:Health_sciences" title="Category:Health sciences">Health sciences</a></li>
<li><a href="/wiki/Category:Medicine" title="Category:Medicine">Medicine</a></li>
<li><a href="/wiki/Category:Neuroscience" title="Category:Neuroscience">Neuroscience</a></li>
<li><a href="/wiki/Category:Zoology" title="Category:Zoology">Zoology</a></li>
<li><i>See also the <a href="#Health_and_fitness">Health and fitness</a> section above</i></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="padding-bottom:0.5em;">
<ul><li><b><a href="/wiki/Category:Earth_sciences" title="Category:Earth sciences">Earth sciences</a></b></li>
<li><a href="/wiki/Category:Atmospheric_sciences" title="Category:Atmospheric sciences">Atmospheric sciences</a></li>
<li><a href="/wiki/Category:Geography" title="Category:Geography">Geography</a></li>
<li><a href="/wiki/Category:Geology" title="Category:Geology">Geology</a></li>
<li><a href="/wiki/Category:Geophysics" title="Category:Geophysics">Geophysics</a></li>
<li><a href="/wiki/Category:Oceanography" title="Category:Oceanography">Oceanography</a></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="padding-bottom:0.5em;">
<ul><li><b><a href="/wiki/Category:Nature" title="Category:Nature">Nature</a></b></li>
<li><a href="/wiki/Category:Animals" title="Category:Animals">Animals</a></li>
<li><a href="/wiki/Category:Natural_environment" title="Category:Natural environment">Environment</a></li>
<li><a href="/wiki/Category:Humans" title="Category:Humans">Humans</a></li>
<li><a href="/wiki/Category:Life" title="Category:Life">Life</a></li>
<li><a href="/wiki/Category:Natural_resources" title="Category:Natural resources">Natural resources</a></li>
<li><a href="/wiki/Category:Plants" title="Category:Plants">Plants</a></li>
<li><a href="/wiki/Category:Pollution" title="Category:Pollution">Pollution</a></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="padding-bottom:0.5em;">
<ul><li><b><a href="/wiki/Category:Physical_sciences" title="Category:Physical sciences">Physical sciences</a></b></li>
<li><a href="/wiki/Category:Astronomy" title="Category:Astronomy">Astronomy</a></li>
<li><a href="/wiki/Category:Chemistry" title="Category:Chemistry">Chemistry</a></li>
<li><a href="/wiki/Category:Climate" title="Category:Climate">Climate</a></li>
<li><a href="/wiki/Category:Physics" title="Category:Physics">Physics</a></li>
<li><a href="/wiki/Category:Space" title="Category:Space">Space</a></li>
<li><a href="/wiki/Category:Universe" title="Category:Universe">Universe</a></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="padding-bottom:0.5em;">
<ul><li><b><a href="/wiki/Category:Scientific_method" title="Category:Scientific method">Scientific method</a></b></li>
<li><a href="/wiki/Category:Scientists" title="Category:Scientists">Scientists</a></li></ul>
</div>""",
            """<figure class="mw-halign-right" typeof="mw:File"><a href="/wiki/File:C_Puzzle.png" class="mw-file-description"><img src="//upload.wikimedia.org/wikipedia/commons/thumb/d/da/C_Puzzle.png/42px-C_Puzzle.png" decoding="async" width="42" height="42" class="mw-file-element" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/d/da/C_Puzzle.png/63px-C_Puzzle.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/d/da/C_Puzzle.png/84px-C_Puzzle.png 2x" data-file-width="150" data-file-height="150"></a><figcaption></figcaption></figure>
<dl><dd><i>Main categories: <a href="/wiki/Category:Technology" title="Category:Technology">Technology</a> and <a href="/wiki/Category:Applied_sciences" title="Category:Applied sciences">Applied sciences</a></i></dd></dl>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="padding: 0.5em 0">
<ul><li><a href="/wiki/Category:Agriculture" title="Category:Agriculture">Agriculture</a></li>
<li><a href="/wiki/Category:Agronomy" title="Category:Agronomy">Agronomy</a></li>
<li><a href="/wiki/Category:Architecture" title="Category:Architecture">Architecture</a></li>
<li><a href="/wiki/Category:Automation" title="Category:Automation">Automation</a></li>
<li><a href="/wiki/Category:Biotechnology" title="Category:Biotechnology">Biotechnology</a></li>
<li><a href="/wiki/Category:Cartography" title="Category:Cartography">Cartography</a></li>
<li><a href="/wiki/Category:Chemical_engineering" title="Category:Chemical engineering">Chemical engineering</a></li>
<li><a href="/wiki/Category:Communication" title="Category:Communication">Communication</a>
<ul><li><a href="/wiki/Category:Media_studies" title="Category:Media studies">Media studies</a></li>
<li><a href="/wiki/Category:Telecommunications" title="Category:Telecommunications">Telecommunications</a></li></ul></li>
<li><a href="/wiki/Category:Construction" title="Category:Construction">Construction</a></li>
<li><a href="/wiki/Category:Control_theory" title="Category:Control theory">Control theory</a></li>
<li><a href="/wiki/Category:Design" title="Category:Design">Design</a></li>
<li><a href="/wiki/Category:Digital_divide" title="Category:Digital divide">Digital divide</a></li>
<li><a href="/wiki/Category:Earthquake_engineering" title="Category:Earthquake engineering">Earthquake engineering</a></li>
<li><a href="/wiki/Category:Energy" title="Category:Energy">Energy</a></li>
<li><a href="/wiki/Category:Ergonomics" title="Category:Ergonomics">Ergonomics</a></li>
<li><a href="/wiki/Category:Firefighting" title="Category:Firefighting">Firefighting</a></li>
<li><a href="/wiki/Category:Fire_prevention" title="Category:Fire prevention">Fire prevention</a></li>
<li><a href="/wiki/Category:Forensic_science" title="Category:Forensic science">Forensic science</a></li>
<li><a href="/wiki/Category:Forestry" title="Category:Forestry">Forestry</a></li>
<li><a href="/wiki/Category:Secondary_sector_of_the_economy" title="Category:Secondary sector of the economy">Industry</a></li>
<li><a href="/wiki/Category:Information_science" title="Category:Information science">Information science</a></li>
<li><a href="/wiki/Category:Internet" title="Category:Internet">Internet</a></li>
<li><a href="/wiki/Category:Management" title="Category:Management">Management</a></li>
<li><a href="/wiki/Category:Manufacturing" title="Category:Manufacturing">Manufacturing</a></li>
<li><a href="/wiki/Category:Marketing" title="Category:Marketing">Marketing</a></li>
<li><a href="/wiki/Category:Medicine" title="Category:Medicine">Medicine</a>
<ul><li><a href="/wiki/Category:Unsolved_problems_in_neuroscience" title="Category:Unsolved problems in neuroscience">Unsolved problems in neuroscience</a></li></ul></li>
<li><a href="/wiki/Category:Metalworking" title="Category:Metalworking">Metalworking</a></li>
<li><a href="/wiki/Category:Microtechnology" title="Category:Microtechnology">Microtechnology</a></li>
<li><a href="/wiki/Category:Military_science" title="Category:Military science">Military science</a></li>
<li><a href="/wiki/Category:Mining" title="Category:Mining">Mining</a></li>
<li><a href="/wiki/Category:Nanotechnology" title="Category:Nanotechnology">Nanotechnology</a></li>
<li><a href="/wiki/Category:Nuclear_technology" title="Category:Nuclear technology">Nuclear technology</a></li>
<li><a href="/wiki/Category:Optics" title="Category:Optics">Optics</a></li>
<li><a href="/wiki/Category:Plumbing" title="Category:Plumbing">Plumbing</a></li>
<li><a href="/wiki/Category:Robotics" title="Category:Robotics">Robotics</a></li>
<li><a href="/wiki/Category:Sound_technology" title="Category:Sound technology">Sound technology</a></li>
<li><a href="/wiki/Category:Technology_forecasting" title="Category:Technology forecasting">Technology forecasting</a></li>
<li><a href="/wiki/Category:Tools" title="Category:Tools">Tools</a></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="padding-bottom:0.5em;">
<ul><li><b><a href="/wiki/Category:Computing" title="Category:Computing">Computing</a></b></li>
<li><a href="/wiki/Category:Apps" title="Category:Apps">Apps</a></li>
<li><a href="/wiki/Category:Artificial_intelligence" title="Category:Artificial intelligence">Artificial intelligence</a></li>
<li><a href="/wiki/Category:Classes_of_computers" title="Category:Classes of computers">Classes of computers</a></li>
<li><a href="/wiki/Category:Computing_by_company" title="Category:Computing by company">Companies</a></li>
<li><a href="/wiki/Category:Computer_architecture" title="Category:Computer architecture">Computer architecture</a></li>
<li><a href="/wiki/Category:Computing_by_computer_model" title="Category:Computing by computer model">Computer model</a></li>
<li><a href="/wiki/Category:Computer_engineering" title="Category:Computer engineering">Computer engineering</a></li>
<li><a href="/wiki/Category:Computer_science" title="Category:Computer science">Computer science</a></li>
<li><a href="/wiki/Category:Computer_security" title="Category:Computer security">Computer security</a></li>
<li><a href="/wiki/Category:Computing_and_society" title="Category:Computing and society">Computing and society</a></li>
<li><a href="/wiki/Category:Computer_data" title="Category:Computer data">Data</a></li>
<li><a href="/wiki/Category:Embedded_systems" title="Category:Embedded systems">Embedded systems</a></li>
<li><a href="/wiki/Category:Free_software" title="Category:Free software">Free software</a></li>
<li><a href="/wiki/Category:Human%E2%80%93computer_interaction" title="Category:Human–computer interaction">Human–computer interaction</a></li>
<li><a href="/wiki/Category:Information_systems" title="Category:Information systems">Information systems</a></li>
<li><a href="/wiki/Category:Information_technology" title="Category:Information technology">Information technology</a></li>
<li><a href="/wiki/Category:Internet" title="Category:Internet">Internet</a></li>
<li><a href="/wiki/Category:Mobile_web" title="Category:Mobile web">Mobile web</a></li>
<li><a href="/wiki/Category:Computer_languages" title="Category:Computer languages">Languages</a></li>
<li><a href="/wiki/Category:Multimedia" title="Category:Multimedia">Multimedia</a></li>
<li><a href="/wiki/Category:Computer_networks" title="Category:Computer networks">Networks <span style="font-size:85%;">(Industrial)</span></a></li>
<li><a href="/wiki/Category:Operating_systems" title="Category:Operating systems">Operating systems</a></li>
<li><a href="/wiki/Category:Computing_platforms" title="Category:Computing platforms">Platforms</a></li>
<li><a href="/wiki/Category:Product_lifecycle_management" title="Category:Product lifecycle management">Product lifecycle management</a></li>
<li><a href="/wiki/Category:Computer_programming" title="Category:Computer programming">Programming</a></li>
<li><a href="/wiki/Category:Real-time_computing" title="Category:Real-time computing">Real-time computing</a></li>
<li><a href="/wiki/Category:Software" title="Category:Software">Software</a></li>
<li><a href="/wiki/Category:Software_engineering" title="Category:Software engineering">Software engineering</a></li>
<li><a href="/wiki/Category:Unsolved_problems_in_computer_science" title="Category:Unsolved problems in computer science">Unsolved problems in computer science</a></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="padding-bottom:0.5em;">
<ul><li><b><a href="/wiki/Category:Electronics" title="Category:Electronics">Electronics</a></b></li>
<li><a href="/wiki/Category:Avionics" title="Category:Avionics">Avionics</a></li>
<li><a href="/wiki/Category:Electronic_circuits" title="Category:Electronic circuits">Circuits</a></li>
<li><a href="/wiki/Category:Electronics_companies" title="Category:Electronics companies">Companies</a></li>
<li><a href="/wiki/Category:Electrical_connectors" title="Category:Electrical connectors">Connectors</a></li>
<li><a href="/wiki/Category:Consumer_electronics" title="Category:Consumer electronics">Consumer electronics</a></li>
<li><a href="/wiki/Category:Digital_electronics" title="Category:Digital electronics">Digital electronics</a></li>
<li><a href="/wiki/Category:Digital_media" title="Category:Digital media">Digital media</a></li>
<li><a href="/wiki/Category:Electrical_components" title="Category:Electrical components">Electrical components</a></li>
<li><a href="/wiki/Category:Electronic_design" title="Category:Electronic design">Electronic design</a></li>
<li><a href="/wiki/Category:Electronics_manufacturing" title="Category:Electronics manufacturing">Electronics manufacturing</a></li>
<li><a href="/wiki/Category:Embedded_systems" title="Category:Embedded systems">Embedded systems</a></li>
<li><a href="/wiki/Category:Integrated_circuits" title="Category:Integrated circuits">Integrated circuits</a></li>
<li><a href="/wiki/Category:Microwave_technology" title="Category:Microwave technology">Microwave technology</a></li>
<li><a href="/wiki/Category:Molecular_electronics" title="Category:Molecular electronics">Molecular electronics</a></li>
<li><a href="/wiki/Category:Water_technology" title="Category:Water technology">Water technology</a></li>
<li><a href="/wiki/Category:Optoelectronics" title="Category:Optoelectronics">Optoelectronics</a></li>
<li><a href="/wiki/Category:Quantum_electronics" title="Category:Quantum electronics">Quantum electronics</a></li>
<li><a href="/wiki/Category:Radio-frequency_identification" title="Category:Radio-frequency identification">Radio-frequency identification <span style="font-size:85%;">RFID</span></a></li>
<li><a href="/wiki/Category:Radio_electronics" title="Category:Radio electronics">Radio electronics</a></li>
<li><a href="/wiki/Category:Semiconductors" title="Category:Semiconductors">Semiconductors</a></li>
<li><a href="/wiki/Category:Signal_cables" title="Category:Signal cables">Signal cables</a></li>
<li><a href="/wiki/Category:Surveillance" title="Category:Surveillance">Surveillance</a></li>
<li><a href="/wiki/Category:Telecommunications" title="Category:Telecommunications">Telecommunications</a></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="padding-bottom:0.5em;">
<ul><li><b><a href="/wiki/Category:Engineering" title="Category:Engineering">Engineering</a></b></li>
<li><a href="/wiki/Category:Aerospace_engineering" title="Category:Aerospace engineering">Aerospace engineering</a></li>
<li><a href="/wiki/Category:Bioengineering" title="Category:Bioengineering">Bioengineering</a></li>
<li><a href="/wiki/Category:Chemical_engineering" title="Category:Chemical engineering">Chemical engineering</a></li>
<li><a href="/wiki/Category:Civil_engineering" title="Category:Civil engineering">Civil engineering</a></li>
<li><a href="/wiki/Category:Electrical_engineering" title="Category:Electrical engineering">Electrical engineering</a></li>
<li><a href="/wiki/Category:Environmental_engineering" title="Category:Environmental engineering">Environmental engineering</a></li>
<li><a href="/wiki/Category:Materials_science" title="Category:Materials science">Materials science</a></li>
<li><a href="/wiki/Category:Mechanical_engineering" title="Category:Mechanical engineering">Mechanical engineering</a></li>
<li><a href="/wiki/Category:Nuclear_technology" title="Category:Nuclear technology">Nuclear technology</a></li>
<li><a href="/wiki/Category:Software_engineering" title="Category:Software engineering">Software engineering</a></li>
<li><a href="/wiki/Category:Structural_engineering" title="Category:Structural engineering">Structural engineering</a></li>
<li><a href="/wiki/Category:Systems_engineering" title="Category:Systems engineering">Systems engineering</a></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="padding-bottom:0.5em;">
<ul><li><b><a href="/wiki/Category:Transport" title="Category:Transport">Transport</a></b></li>
<li><a href="/wiki/Category:Transport_by_country" title="Category:Transport by country">By country</a></li>
<li><a href="/wiki/Category:Aviation" title="Category:Aviation">Aviation</a></li>
<li><a href="/wiki/Category:Cars" title="Category:Cars">Cars</a></li>
<li><a href="/wiki/Category:Cycling" title="Category:Cycling">Cycling</a></li>
<li><a href="/wiki/Category:Public_transport" title="Category:Public transport">Public transport</a></li>
<li><a href="/wiki/Category:Rail_transport" title="Category:Rail transport">Rail transport</a></li>
<li><a href="/wiki/Category:Road_transport" title="Category:Road transport">Road transport</a></li>
<li><a href="/wiki/Category:Shipping" title="Category:Shipping">Shipping</a></li>
<li><a href="/wiki/Category:Spaceflight" title="Category:Spaceflight">Spaceflight</a></li>
<li><a href="/wiki/Category:Vehicles" title="Category:Vehicles">Vehicles</a></li>
<li><a href="/wiki/Category:Water_transport" title="Category:Water transport">Water transport</a></li></ul>
</div>
<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1129693374"><div class="hlist" style="padding-bottom:0.5em;">
<ul><li><i>See also: <a href="/wiki/Category:Technology_timelines" title="Category:Technology timelines">Technology timelines</a></i></li></ul>
</div>"""
            ]


def find_cats(a):
    pattern = re.compile(r'<a href="([^"]+)" title="([^"]+)">([^<]+)</a>')
    matches = pattern.findall(a)
    finds = []
    for x in matches:
        finds.append(x[2].replace(" ", "_"))
    return finds


if __name__ == "__main__":
    results = []
    for raw in raw_html:
        results.extend(find_cats(raw))

    print(results)
    print(len(results))
    with open("data/roots.json", "w") as f:
        json.dump(results, f)
