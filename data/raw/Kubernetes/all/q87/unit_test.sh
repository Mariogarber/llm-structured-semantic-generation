kubectl apply -f labeled_code.yaml
GROUP=$(kubectl get crd myresources.example.com -o=jsonpath='{.spec.group}')
VERSION=$(kubectl get crd myresources.example.com -o=jsonpath='{.spec.versions[0].name}')
PLURAL=$(kubectl get crd myresources.example.com -o=jsonpath='{.spec.names.plural}')

[ "$GROUP" = "example.com" ] && \
[ "$VERSION" = "v1" ] && \
[ "$PLURAL" = "myresources" ] && \
echo cloudeval_unit_test_passed
