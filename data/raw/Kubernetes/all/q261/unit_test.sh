kubectl apply -f labeled_code.yaml

kubectl get serviceaccount | grep -q "simple-sa" && 
echo cloudeval_unit_test_passed
