To update the documentation ...
 - git branch gh-pages
 - git pull
 - cd docs
 - make
 - cd ..
 - cp -pr docs/_build/html/* .
 - git add .
 - git commit -m "updated docs"
 - git push
