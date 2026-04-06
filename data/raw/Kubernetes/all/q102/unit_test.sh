kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete jobs/hello
kubectl get job hello -o=jsonpath='{.spec.template.spec.restartPolicy}' | grep "OnFailure" && echo cloudeval_unit_test_passed
# Stackoverflow https://stackoverflow.com/questions/40530946/what-is-the-difference-between-always-and-on-failure-for-kubernetes-restart-poli