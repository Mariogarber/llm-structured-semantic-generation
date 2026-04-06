kubectl apply -f labeled_code.yaml
# kubectl wait replicaset --all --for=condition=Running --timeout=20s
# make sure:
# 3 pods are Running
for i in {1..120}; do
    kubectl describe rs/frontend | awk -v RS='' '\
    /3 Running/ \
    {print "cloudeval_unit_test_passed"; found=1} \
    END {if (found!=1) {exit 1}}' && break
    sleep 1
done