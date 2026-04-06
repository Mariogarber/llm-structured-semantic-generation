kubectl create ns development
kubectl create secret generic test-secret --from-literal=password=my-password -n development
kubectl apply -f labeled_code.yaml

cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: secret-reader
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list"]
EOF

cleanup() {
  kubectl delete secret test-secret -n development
  kubectl delete ns development
  kubectl delete clusterrole secret-reader
  kubectl delete -f labeled_code.yaml
}

namespace=$(kubectl get rolebinding read-secrets -n development -o=jsonpath='{.metadata.namespace}')
subject_name=$(kubectl get rolebinding read-secrets -n development -o=jsonpath='{.subjects[0].name}')
role_ref_name=$(kubectl get rolebinding read-secrets -n development -o=jsonpath='{.roleRef.name}')

if [[ $namespace == "development" && $subject_name == "dave" && $role_ref_name == "secret-reader" ]]; then
    echo cloudeval_unit_test_passed
else
    echo "Test failed"
fi

cleanup