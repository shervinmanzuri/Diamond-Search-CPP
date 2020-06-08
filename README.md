Diamond Search Block Matching
=================

Zhu et al.'s [diamond search algorithm](https://ieeexplore.ieee.org/document/821744) for motion vector estimation. Written in C++.


Requirements
--------------------

- OpenCV 4
- C++ 11


Inputs
--------------------
 
- CV Mat current image and reference image
- Integer macro-block size
- Cost function (e.g. PSNR or MAD)
- Integer skipping, how many n blocks to skip. This leads to a high performance increase at negligible cost of accuracy.


Outputs
--------------------

- Diamond search computations (can omit).
- Block motion vectors.


Future Work
--------------------

- Better looking readme file.
- Code refactoring.
- Merge Earthrealm into Outworld.
