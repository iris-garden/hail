/* global constants */
const CLIENT_ID = "894037887906-km397kr4l6j1pkrueph0kihgcvtmbmh2.apps.googleusercontent.com"


/* DOM helpers */
const getEle = (idOrEle) =>
  typeof idOrEle === "string" ? document.getElementById(idOrEle) : idOrEle

const withEle = (idOrEle, callback) => {
  const ele = getEle(idOrEle)
  callback(ele)
  return ele
}

const updateEle = (idOrEle, props) =>
  withEle(idOrEle, (ele) => {
    Object.assign(ele, props)
  })

const appendToEle = (idOrEle, childType, childProps) =>
  withEle(idOrEle, (ele) => {
    const childEle = document.createElement(childType)
    updateEle(childEle, childProps)
    ele.appendChild(childEle)
  })

const hideEle = (idOrEle) =>
  updateEle(idOrEle, { className: "display-none" })

const showEle = (idOrEle, { className = "", ...props } = {}) =>
  updateEle(idOrEle, { className, ...props })


/* page-specific helpers */
const [
  messageEle,
  selectProjectLabelEle,
  selectProjectEle,
  getOrCreateBucketEle,
] = [
  "message",
  "select-project-label",
  "select-project",
  "get-or-create-bucket",
].map(getEle)

const hideMessage = () =>
  hideEle(messageEle)

const showMessage = (message, error) =>
  showEle(messageEle, { innerHTML: error !== undefined ? `${message}: ${error}` : message })

const showGapiError = () => showMessage("The Google API client failed to load")


/* main code (run when both API wrappers have loaded and the user is signed in) */
const main = () => {
  hideMessage()
  gapi.client.cloudresourcemanager.projects.list({})
    .then(response => {
      showEle(selectProjectLabelEle)
      const selectProject = showEle(selectProjectEle)
      response.result.projects.forEach(({ projectId }) =>
        appendToEle(selectProject, "option", { innerHTML: projectId, value: projectId })
      )
      showEle(getOrCreateBucketEle).addEventListener("click", () => {
        const projectName = updateEle(selectProject, { disabled: true }).value
        gapi.client.people.people.get({
          resourceName: "people/me",
          "requestMask.includeField": "person.emailAddresses",
        }).then(person => {
          let emailIdx
          const { value: emailAddress } = person.result.emailAddresses.find(emailAddress => {
            const idx = emailAddress.value.indexOf("@broadinstitute.org")
            const found = idx !== -1
            if (found) {
              emailIdx = idx
            }
            return found
          })
          if (emailIdx !== undefined && emailAddress !== undefined) {
            const bucketName = `${projectName}-${emailAddress.slice(0, emailIdx)}-qob`
            gapi.client.storage.buckets.get({
              bucket: bucketName,
              userProject: projectName,
            }).then(bucket => 
              showMessage(`Bucket ${bucketName} exists.`)
              // TODO set cookie and redirect to jupyterlite
            ).catch(() =>
              gapi.client.storage.buckets.insert({
                project: projectName,
                resource: { name: bucketName },
              }).then(bucket => {
                showMessage(`Bucket ${bucketName} was created.`)
                // TODO set cookie and redirect to jupyterlite
              }).catch(error => showMessage(`Unable to create bucket ${bucketName}`, error))
            )
          } else {
            showMessage("User has no @broadinstitute.org email address.")
          }
        }).catch(error => showMessage("Unable to get user information", error))
        selectProject.value
      })
    })
    .catch(error => showMessage("Unable to list Google Cloud Projects", error))
}


/* API wrapper loaders and sign-in flow */
try {
  gapi.load("client", {
    callback: () => {
      gapi.client.init({}).then(() => {
        Promise.all([
          "https://cloudresourcemanager.googleapis.com/$discovery/rest?version=v1",
          "https://people.googleapis.com/$discovery/rest?version=v1",
          "https://storage.googleapis.com/$discovery/rest?version=v1",
        ].map(api => gapi.client.load(api))).then(() => {
          try {
            const tokenClient = google.accounts.oauth2.initTokenClient({
              client_id: CLIENT_ID,
              scope: [
                "cloud-platform",
                "userinfo.profile",
                "devstorage.full_control",
              ].map(scope => `https://www.googleapis.com/auth/${scope}`).join(" "),
              prompt: "consent",
              callback: response => {
                if (response.error === undefined) {
                  main()
                } else {
                  showMessage("Google Identity Services failed to obtain token", response.error)
                }
              },
            })
            showMessage("Waiting for you to sign in with Google in another tab...")
            tokenClient.requestAccessToken()
          } catch (error) {
            showMessage("Google Identity Services failed to load", error)
          }
        })
      })
    },
    onerror: showGapiError,
  })
} catch {
  showGapiError()
}
