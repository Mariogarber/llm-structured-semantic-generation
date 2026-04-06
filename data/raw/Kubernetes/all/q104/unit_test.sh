kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete job/indexed-job --timeout=60s
pod0=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep 'indexed-job-0')
pod1=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep 'indexed-job-1')
pod2=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep 'indexed-job-2')
pod3=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep 'indexed-job-3')
pod4=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep 'indexed-job-4')

[ "$(kubectl logs $pod0 worker)" = "oof" ] && \
[ "$(kubectl logs $pod1 worker)" = "rab" ] && \
[ "$(kubectl logs $pod2 worker)" = "zab" ] && \
[ "$(kubectl logs $pod3 worker)" = "xuq" ] && \
[ "$(kubectl logs $pod4 worker)" = "zyx" ] && \
echo cloudeval_unit_test_passed
