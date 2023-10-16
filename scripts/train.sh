git clone https://github.com/leifu1128/RedunMin.git
sh RedunMin/scripts/install_deps.sh
python RedunMin/train.py

curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env' -H 'Metadata-Flavor: Google' > /tmp/tpu-env # store the metadata
NODE_ID=$(grep '^NODE_ID' /tmp/tpu-env | cut -d "'" -f 2) # get Node ID
WORKER_ID=$(grep '^WORKER_ID' /tmp/tpu-env | cut -d "'" -f 2) # get Worker ID
ZONE=$(grep '^ZONE' /tmp/tpu-env | cut -d "'" -f 2) # get VM zone
if [ $WORKER_ID==0 ]; then
    gcloud alpha compute tpus tpu-vm delete $NODE_ID --zone=$ZONE --quiet; # delete the node
fi