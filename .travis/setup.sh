#!/bin/bash

#setup travis-ci configuration basing one the being-built branch

if [[ $TRAVIS_BRANCH == 'master' ]] ; then
    export DEPLOY_HTML_DIR=docs
elif [[ $TRAVIS_BRANCH == 'develop' ]] ; then
    export DEPLOY_HTML_DIR=docs/develop
elif [[ $TRAVIS_BRANCH =~ ^v[0-9.]+$ ]]; then
    export DEPLOY_HTML_DIR=docs/${TRAVIS_BRANCH:1}
elif [[ $TRAVIS_BRANCH =~ ^[0-9.]+\.x$ || $TRAVIS_BRANCH =~ ^releases\/[0-9.]+$ ]]; then
    export DEPLOY_HTML_DIR=$TRAVIS_BRANCH
fi
