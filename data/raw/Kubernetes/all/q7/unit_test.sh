kubectl apply -f labeled_code.yaml
i=0; while [ $i -lt 60 ] && [ $(kubectl get jobs | grep example | wc -l) -lt 1 ]; do sleep 1; i=$((i+1));echo $i; done
sleep 70
pods=$(kubectl get pods -o=jsonpath='{.items[0].metadata.name}')
kubectl logs $pods | grep "OK" && echo cloudeval_unit_test_passed_1
kubectl get pods | grep "example" | wc -l | grep 1 && echo cloudeval_unit_test_passed_2
kubectl get pods