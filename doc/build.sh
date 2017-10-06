#!/bin/sh -e
# Prepare documentation for release via GitHub Pages.
#
# N.B., this script changes the current branch! Generally Git will not
# allow `git checkout` to destroy unsaved work in the working tree.
# However, if you are still concerned, try `git stash -u`.
#
# After running this script, it remains to `git push origin gh-pages`.
#
# After pushing, you must manually change back to the branch that you
# were using before this script.

git checkout -b gh-pages origin/gh-pages || git checkout gh-pages
cp -f tutorial.md ../index.md
git reset HEAD -- ..
git add ../index.md
if git commit -m 'Release documentation via doc/build.sh'; then
    echo "Done. When you are ready, deploy using:"
    echo "git push origin gh-pages"
else
    echo "No changes since last release."
fi
echo "You must manually change back to the branch that you were using"
echo "before running this script."
