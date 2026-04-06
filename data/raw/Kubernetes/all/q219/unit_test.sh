kubectl apply -f labeled_code.yaml
ROLE_NAME=$(kubectl get rolebinding my-rb -o=jsonpath='{.roleRef.name}')
SA_NAME=$(kubectl get rolebinding my-rb -o=jsonpath='{.subjects[0].name}')

[ "$ROLE_NAME" = "read" ] && \
[ "$SA_NAME" = "read-sa" ] && \
echo cloudeval_unit_test_passed
