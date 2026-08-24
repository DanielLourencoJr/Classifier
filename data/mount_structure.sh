cd mine 
mine=$(ls -1 | wc -l)

cd ../others
others=$(ls -1 | wc -l)

cd ..
mkdir -p train/mine train/others test/mine test/others

cd mine
mine_train_n=$(( mine * 80 / 100 ))

shuffled=$(ls -1 | shuf)

train_files=$(echo "$shuffled" | head -n "$mine_train_n")
test_files=$(echo "$shuffled" | tail -n +"$((mine_train_n + 1))")

echo "$train_files" | xargs -I{} cp {} ../train/mine/
echo "$test_files" | xargs -I{} cp {} ../test/mine/

cd ../others
others_train_n=$(( others * 80 / 100 ))

shuffled=$(ls -1 | shuf)

train_files=$(echo "$shuffled" | head -n "$others_train_n")
test_files=$(echo "$shuffled" | tail -n +"$((others_train_n + 1))")

echo "$train_files" | xargs -I{} cp {} ../train/others/
echo "$test_files" | xargs -I{} cp {} ../test/others/
