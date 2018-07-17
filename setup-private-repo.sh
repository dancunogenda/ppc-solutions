remotes=`git remote | wc -l`
if [ $remotes -gt 1 ]
then
echo "Looks like you have already run this script once. Doing nothing."
exit
fi

echo "Please enter your GitHub username"
read account
echo "Setting up account '$account'. Unless you have setup SSH keys for GitHub, you will be prompted your GitHub password"

git remote rename origin upstream
git remote add origin https://github.com/ICS-E4020/solutions-$account.git
git push -u origin master

