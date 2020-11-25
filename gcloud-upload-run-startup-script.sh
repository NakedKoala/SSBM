gcloud beta compute scp ./gcloud-startup-script.sh $1:~/ && \
gcloud beta compute ssh --zone "us-east1-c" $1 --project "cs486-g84-melee" --command="~/gcloud-startup-script.sh"
