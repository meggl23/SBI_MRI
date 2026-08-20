Preliminary version of the SBI + dwMRI package. 
-- 
Models are built in a modular way so that compartments can be added/removed at will. Currently we have
- DTI
- Cylinders
- Spheres
- Free water

While some validation has been done, please talk to me before you use it in any "publication" level setting. I have also included the environment (_environment.yml_). You can install the environment then with:

```conda env create -f environment.yml```

Also there is an example in-vivo dataset you can use to do some testing (also useful for the tutorial). This data is downloaded in the tutorial, but if you are really keen you can also download it [here](https://www.dropbox.com/scl/fi/xki53zdriavev4rt22wjz/data.zip?rlkey=ejfw5malh99l98sv747jg273c&dl=1).

Please if you have any wishes on what we should implement let me know - within reason. Also, if you are using this and you find any bugs/things that don't work please let me know as well. Try and document the bug as much as possible and then send me the details.

Now if you want to make changes to the code you are welcome to - *within reason*. Before making big changes please create your own branch and change that there, that means the main part of the code doesn't get changed for everyone else. Once you are happy with your changes and I can verify them I will think about including them in the main branch. There will be no more stupid fragmentation in the lab.
